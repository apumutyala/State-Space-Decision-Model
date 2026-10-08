"""Train/eval equivalence of the jrt-rwkv-g1-v2 template on real m2 training records (CPU, tokenizer only, no model).

For every (record, question) row:
  train side   = train.data.items_from_records(..., template="jrt-rwkv-g1-v2", layout="g1v2") with the cell's own encoder
                 (train.run_cell.parent_encoder("rwkv7", <tokenizer dir>))
  readout side = decide.backends.rwkv_g1_v2.G1V2Backend._questions (the readout's own prompt + code-token code),
                 on a model-free plugin whose encode is the readout plugin's (hf_readout.load_rwkv7_hf) encode body
and it asserts
  ids     : row.ids == the readout's prompt ids ([0] + whole-string tokens)
  options : row.option_ids == the readout's option order
  coverage: every request field's text (state, every question's id + instructions, every option id + description;
            strings blank-line-collapsed as the v2 renderer does, non-strings as canonical JSON or by every string leaf)
            appears in the decoded training prompt -- catches a renderer that drops a field (e.g. dict instructions)
  target  : the train target (row.continuations[label], label = canonical index of the gold) == [readout code token of
            the gold], and row.answer_pos == the position the readout reads (len(prompt) - 1)

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_g1v2_train_equivalence.py \
        --records <records.jsonl.gz> --tokenizer-dir <RWKV7-G1k snapshot with tokenizer.json> [--n-small 120 --n-large 80 --n-multi 30]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pick_records(path: str, n_small: int, n_large: int, n_multi: int) -> list[dict]:
    """First n_small single-question records with <= 16 options, first n_large with some question > 16, and first
    n_multi multi-question records (several rows per record: each question renders its own re-read)."""
    small, large, multi = [], [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if len(small) >= n_small and len(large) >= n_large and len(multi) >= n_multi:
                break
            rec = json.loads(line)
            ks = [len(q["criteria"]) if q["type"] != "noul" else 2 for q in rec["questions"].values()]
            bucket, cap = ((multi, n_multi) if len(ks) > 1 else (large, n_large) if max(ks) > 16
                           else (small, n_small))
            if len(bucket) < cap:
                bucket.append(rec)
    return small + large + multi


def _leaves(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _leaves(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _leaves(x)


def uncovered_fields(req: dict, text: str) -> list[str]:
    """Request fields whose text is missing from the prompt ``text``."""
    from decide.prompt import g1_clean
    from decide.schema import render

    def covered(v) -> bool:
        if v is None or v == "" or v == {} or v == []:
            return True
        if isinstance(v, str):
            return g1_clean(v).strip() in text
        if g1_clean(render(v)) in text:
            return True
        return all(g1_clean(x).strip() in text or g1_clean(render(x)[1:-1]) in text for x in _leaves(v) if x.strip())

    miss = [] if covered(req["state"]) else ["state"]
    for qid, q in req["questions"].items():
        if not covered(qid):
            miss.append(f"{qid}:id")
        if not covered(q.get("instructions")):
            miss.append(f"{qid}:instructions")
        crit = q.get("criteria")
        if isinstance(crit, dict):
            for oid, desc in crit.items():
                if not covered(str(oid)):
                    miss.append(f"{qid}:option_id:{oid}")
                if not covered(desc):
                    miss.append(f"{qid}:option_desc:{oid}")
        elif isinstance(crit, list):
            for i, desc in enumerate(crit):
                if not covered(desc):
                    miss.append(f"{qid}:level:{i}")
    return miss


def dict_instruction_record() -> dict:
    """Synthetic record with dict-valued instructions (the ToolRet / BRIGHT shape the v2 renderer must not drop)."""
    return {"id": "synthetic:dict-instructions", "state": "Retrieve the tool that converts currencies.",
            "questions": {"relevant": {"type": "choice",
                                       "instructions": {"task": "Is this tool relevant to the query?",
                                                        "candidate": "fx_convert: converts an amount between two "
                                                                     "currencies at the live rate"},
                                       "criteria": {"no": None, "yes": None}}},
            "labels": {"relevant": "yes"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--n-small", type=int, default=120)
    ap.add_argument("--n-large", type=int, default=80)
    ap.add_argument("--n-multi", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from decide.backends.rwkv_g1_v2 import BOS_ID, G1V2Backend
    from decide.backends.rwkv_g1_filtered_codes import load_frozen_table
    from decide.prompt import build_prompt_g1v2, render_g1v2
    from decide.untrusted import ControlTokenError
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
    from train.data import items_from_records
    from train.guard import record_request
    from train.run_cell import parent_encoder

    enc = parent_encoder("rwkv7", a.tokenizer_dir)  # train side: the cell's encoder
    tok = Rwkv7Tokenizer.from_pretrained(a.tokenizer_dir)
    special = frozenset(int(i) for i in tok.all_special_ids) | {0}

    def readout_encode(text: str, trusted: bool = True) -> list[int]:  # decide.backends.hf_readout.load_rwkv7_hf
        ids = tok(text, add_special_tokens=False).input_ids
        if not trusted and special.intersection(ids):
            raise ControlTokenError("untrusted text encodes to an RWKV special token (<|endoftext|>)")
        return ids

    ro = G1V2Backend.__new__(G1V2Backend)  # the readout's own _questions, without loading a model
    ro.plugin = SimpleNamespace(encode=readout_encode)
    ro.codes = load_frozen_table()
    ro.code_table = [c.code for c in ro.codes]
    ro._full_memo = {}
    ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"  # the backend's own F4 proof, same tokenizer
    base = readout_encode(ctx, True)
    assert all(readout_encode(ctx + " " + c.code, True) == base + [c.token_id] for c in ro.codes)

    # (k > 255 never reaches a row: decide.schema.validate_legacy caps options at 255 = the code table size)
    records = pick_records(a.records, a.n_small, a.n_large, a.n_multi) + [dict_instruction_record()]
    n_rows, mism, by_k, refused = 0, [], {"<=16": 0, "17-255": 0, ">255": 0}, 0
    span_bad, empty_state, recollapsed = 0, 0, 0
    for rec in records:
        req = record_request(rec)
        try:
            items = items_from_records([rec], enc, layout="g1v2", template="jrt-rwkv-g1-v2")
        except ControlTokenError:
            refused += 1
            continue
        qs = {q["qid"]: q for q in ro._questions(req)}
        for it in items:
            qid = it.id.rsplit("/", 1)[1]
            r, q = it.row, qs[qid]
            k = r.n_options
            by_k["<=16" if k <= 16 else ("17-255" if k <= 255 else ">255")] += 1
            n_rows += 1
            gold = r.option_ids[it.label]
            empty_state += r.meta["evidence"] == "reread_instructions"
            recollapsed += "".join(t for t, _ in build_prompt_g1v2(req, qid, ro.code_table)) != render_g1v2(
                req, qid, ro.code_table)
            errs = []
            if r.ids != q["prompt"]:
                errs.append("ids")
            if r.option_ids != q["options"]:
                errs.append("options")
            if r.answer_pos != len(q["prompt"]) - 1 or r.ids[0] != BOS_ID:
                errs.append("answer_pos")
            if q["coded"]:
                ro_gold = q["tokens"][q["options"].index(gold)]
                if r.continuations[it.label] != [ro_gold] or r.continuations != [[t] for t in q["tokens"]]:
                    errs.append("target")
            elif r.continuations[it.label] != readout_encode(" " + gold + "\n", True):
                errs.append("target_fallback")
            miss = uncovered_fields(req, tok.decode(r.ids[1:]))
            if miss:
                errs.append("coverage:" + ",".join(miss[:5]))
            if errs:
                mism.append({"id": it.id, "k": k, "errors": errs})
            # informational span sanity: state span inside ids, the target field span decodes to its "FIELD" block
            a0, b0 = r.state_span
            fa, fb = r.field_span
            if not (0 < a0 < b0 <= len(r.ids)) or "FIELD" not in tok.decode(r.ids[fa:fb]):
                span_bad += 1
    res = {"records": a.records, "n_records": len(records), "n_rows": n_rows, "rows_by_k": by_k,
           "refused_records": refused, "mismatches": len(mism), "mismatch_examples": mism[:10],
           "span_sanity_failures": span_bad,
           "rows_empty_state": empty_state, "rows_final_blank_line_recollapse": recollapsed}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not mism and n_rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
