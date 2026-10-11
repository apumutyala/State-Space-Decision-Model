"""train == readout token equivalence (and a T / K census) of the Q9-AF v2 v3w cell, on the B1 100k-v2 training draw.

For each sampled training record (``train.run_cell.to_train_record``, as the cell trains it) and question:
  train   = ``train.data.items_from_records`` with the cell's template / layout (``heads.layout.build_row`` -> the
            generic v3w spec's ``train_row``: the path ``train.run_cell`` trains on)
  readout = ``decide.backends.v3w_generic.V3WGenericBackend._questions`` on a model-free plugin whose ``encode`` is the
            snapshot's HF tokenizer (``transformers.AutoTokenizer``, as ``convert.readout.load_converted_plugin``), so
            the backend's own plugin-tokenizer proofs run too
and requires: train ids == readout prompt ids; option order equal; continuations == the scored tokens ([t] per option
for coded / word-id questions, the multi-token " <id>\\n" continuations for k > code table); the answer read at the last
prompt position; the gold option among the options. Strata: K <= 16, K 17-255, K > 255, word-id questions, dict
(non-string) instructions. Census (``--census N`` records): rows by K, T (len ids) quantiles, the share the cell's
filters keep (train_max_k, max T of the ladder).

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_q9af_v3w_equivalence.py \
        --records <ssdm-data>/m2/build/b1-draw100k-v2/records.jsonl.gz --tokenizer-dir <Qwen3.5-9B tokenizer dir> \
        --cell jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json --min-rows 700 --out report.json
    ... --print-pins   the spec identity sha256 (decide.v3w_generic.PINNED_SPEC_SHA256)
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def model_free_plugin(tok_dir: str, max_context: int = 262144):
    from transformers import AutoTokenizer

    from decide.untrusted import UntrustedEncoder

    tok = AutoTokenizer.from_pretrained(tok_dir)
    untrusted = UntrustedEncoder(tok)

    def encode(text: str, trusted: bool = True) -> list[int]:  # convert.readout.load_converted_plugin's encode
        return tok(text, add_special_tokens=False).input_ids if trusted else untrusted.encode(text)

    return SimpleNamespace(family="qwen3.5", model_id="model-free", snapshot=None, encode=encode, forward=None,
                           special_ids=untrusted.control_ids, files={}, max_context=max_context, dtype="bfloat16",
                           device="cpu", kernel_stack={}, state_ops=object(), fork_state_token_budget=None,
                           notes={}, untrusted_encoder_id="model-free"), tok


def stratum(row, q, request) -> list[str]:
    k = len(q["options"])
    out = ["k<=16" if k <= 16 else ("k17-255" if k <= 255 else "k>255")]
    if q.get("words"):
        out.append("words")
    ins = request["questions"][q["qid"]].get("instructions")
    if ins is not None and not isinstance(ins, str):
        out.append("dict_instructions")
    return out


def check_row(item, q) -> list[str]:
    row = item.row
    errs = []
    if list(row.ids) != list(q["prompt"]):
        n = next((i for i, (a, b) in enumerate(zip(row.ids, q["prompt"])) if a != b), min(len(row.ids), len(q["prompt"])))
        errs.append(f"ids differ at {n} (train {len(row.ids)} vs readout {len(q['prompt'])})")
    if list(row.option_ids) != list(q["options"]):
        errs.append("option order")
    want = [[t] for t in q["tokens"]] if q["coded"] else [list(c) for c in q["conts"]]
    got = [list(c) for c in row.continuations]
    if got != want:
        errs.append("continuations != scored tokens")
    if q["coded"] and any(t is None for t in q["tokens"]):
        errs.append("a scored token is None")
    ap = getattr(row, "answer_pos", None)
    if ap is not None and ap != len(row.ids) - 1:
        errs.append(f"answer_pos {ap} != len(prompt) - 1")
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--cell", default=str(ROOT / "jobs/m2_first_cell/m2_first_cell_rwkv7_q9af_v2_v3w.json"))
    ap.add_argument("--sample", type=int, default=600, help="records for the equivalence check (random, seeded)")
    ap.add_argument("--min-rows", type=int, default=700)
    ap.add_argument("--min-per-stratum", type=int, default=20)
    ap.add_argument("--census", type=int, default=0, help="records for the T / K census (0 = skip)")
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--print-pins", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    from decide.backends.v3w_generic import V3WGenericBackend
    from decide.v3w_generic import bind_template, make_v3w, template_id
    from train.data import items_from_records
    from train.run_cell import keep_item, to_train_record, train_max_k

    if a.print_pins:
        v = make_v3w("qwen3.5-9b", tokenizer_dir=a.tokenizer_dir)
        print(json.dumps({template_id("qwen3.5-9b"): v.identity()["sha256"]}))
        return 0
    cfg = json.loads(Path(a.cell).read_text(encoding="utf-8"))
    tpl, lay = cfg["template"], cfg["layout"]
    spec = bind_template(tpl, a.tokenizer_dir)
    plug, tok = model_free_plugin(a.tokenizer_dir)
    backend = V3WGenericBackend(plug, spec)  # runs the plugin-tokenizer == spec proofs
    from heads.layout import hf_segment_encoder

    enc = hf_segment_encoder(tok)  # the training encoder run_cell uses for parent q9af (ignored by the generic rows)
    t0 = time.time()
    with gzip.open(a.records, "rt", encoding="utf-8") as f:
        lines = f.readlines()
    rng = random.Random(a.seed)
    order = list(range(len(lines)))
    rng.shuffle(order)
    # equivalence: a seeded random sample + every record of the rare strata found in the next 20k (K > 16, dicts)
    pick, big_k, dicts = order[: a.sample], [], []
    for i in order[a.sample: a.sample + 60000]:
        r = json.loads(lines[i])
        ks = [len((q.get("criteria") or {})) for q in r["questions"].values()]
        if any(not isinstance(q.get("instructions"), str) for q in r["questions"].values()):
            dicts.append(i)
        elif any(k > 16 for k in ks):
            big_k.append(i)
    pick += big_k[:250] + dicts[:250]
    strata, mism, n_rows, refused, examples = Counter(), [], 0, Counter(), []
    for i in pick:
        rec = to_train_record(json.loads(lines[i]))
        try:
            items = items_from_records([rec], enc, layout=lay, template=tpl)
        except ValueError as e:
            refused[type(e).__name__ + ": " + str(e)[:60]] += 1
            continue
        from train.guard import record_request

        req = record_request(rec)
        qs = {q["qid"]: q for q in backend._questions(req)}
        for it in items:
            qid = it.id.rsplit("/", 1)[1]
            q = qs[qid]
            n_rows += 1
            for s in stratum(it.row, q, req):
                strata[s] += 1
            e = check_row(it, q)
            if e:
                mism.append({"id": it.id, "errors": e})
            elif len(examples) < 3:
                examples.append({"id": it.id, "k": len(q["options"]), "T": len(it.row.ids), "words": q["words"]})
    rep = {"template": tpl, "layout": lay, "spec_identity": spec.identity(), "records": a.records,
           "records_checked": len(pick), "rows": n_rows, "mismatches": len(mism), "mismatch_examples": mism[:10],
           "strata": dict(strata), "refused_records": dict(refused), "examples": examples,
           "backend_revision": backend.revision()["revision_id"]}
    ok = (not mism and n_rows >= a.min_rows
          and all(strata[s] >= a.min_per_stratum for s in ("k<=16", "k17-255", "words", "dict_instructions")))
    if a.census:
        kmax, tmax = train_max_k(cfg), max(cfg["t_ladder"])
        ks, ts, kept, tot = Counter(), [], 0, 0
        for i in order[: a.census]:
            rec = to_train_record(json.loads(lines[i]))
            try:
                items = items_from_records([rec], enc, layout=lay, template=tpl)
            except ValueError:
                continue
            for it in items:
                tot += 1
                k = len(it.row.option_ids)
                ks["k<=16" if k <= 16 else ("k17-255" if k <= 255 else "k>255")] += 1
                ts.append(len(it.row.ids))
                kept += keep_item(it.row, kmax, tmax)
        ts.sort()
        qn = lambda p: ts[min(len(ts) - 1, int(p * len(ts)))] if ts else None  # noqa: E731
        rep["census"] = {"records": a.census, "items": tot, "kept_by_cell_filter": kept, "train_max_k": kmax,
                         "max_t": tmax, "by_k": dict(ks), "T_mean": round(sum(ts) / max(1, len(ts)), 1),
                         "T_p50": qn(0.5), "T_p90": qn(0.9), "T_p99": qn(0.99), "T_max": ts[-1] if ts else None,
                         "over_max_t": sum(t > tmax for t in ts)}
    rep["ok"] = ok
    rep["seconds"] = round(time.time() - t0, 1)
    s = json.dumps(rep, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s, encoding="utf-8")
    print(s)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
