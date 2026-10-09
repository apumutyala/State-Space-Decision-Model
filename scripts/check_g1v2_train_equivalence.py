"""Train/eval equivalence of the jrt-rwkv-g1-v2 template on real m2 training records (CPU, tokenizer only, no model).

For every (record, question) row:
  train side   = train.data.items_from_records(..., template=<--template>, layout="g1v2") with the cell's own encoder
                 (train.run_cell.parent_encoder("rwkv7", <tokenizer dir>))
  readout side = the template's readout backend's own _questions: jrt-rwkv-g1-v2 -> rwkv_g1_v2.G1V2Backend,
                 jrt-rwkv-g1-v2f -> rwkv_g1_v3.G1V3Backend(mode="v2f") (the readout's own prompt + code-token code),
                 jrt-rwkv-g1-v3w -> rwkv_g1_v3.G1V3Backend(mode="words") (v3 code table, code key + its budget ladder,
                 word ids scored by the first token of " <id>"),
                 on a model-free plugin whose encode is the readout plugin's (hf_readout.load_rwkv7_hf) encode body
and it asserts
  ids     : row.ids == the readout's prompt ids ([0] + whole-string tokens)
  options : row.option_ids == the readout's option order
  coverage: every request field's text (state, every question's id + instructions, every option id + description;
            strings blank-line-collapsed as the v2 renderer does, non-strings as canonical JSON or by every string leaf)
            appears in the decoded training prompt -- catches a renderer that drops a field (e.g. dict instructions)
  v3w     : the row's word-id flag and code-key label width == the readout's (q["words"], q["catalog_chars"])
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


def pick_stratified(path: str, per: int) -> list[dict]:
    """The first ``per`` records of every stratum (family, question type, option-count bucket <=16 / 17-255,
    word-like option ids (non-opaque, decide.prompt_g1v3.is_opaque_id), dict-valued instructions, multi-question)
    over the WHOLE file (the file is family-sorted), so every family and id style is covered."""
    from decide.prompt_g1v3 import is_opaque_id

    seen: dict[tuple, int] = {}
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            keys = set()
            for q in rec["questions"].values():
                crit = q.get("criteria")
                ids = [str(o) for o in crit] if isinstance(crit, dict) else []
                k = len(crit) if crit is not None else 2
                instr = q.get("instructions")
                keys.add((rec.get("family"), q["type"], "<=16" if k <= 16 else "17-255",
                          bool(ids) and not any(is_opaque_id(o) for o in ids),
                          instr is not None and not isinstance(instr, str), len(rec["questions"]) > 1))
            if any(seen.get(key, 0) < per for key in keys):
                out.append(rec)
                for key in keys:
                    seen[key] = seen.get(key, 0) + 1
    return out


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


def word_id_records() -> list[dict]:
    """Synthetic word-id families absent from the 150k training set (fact verification SUPPORTED/REFUTED/NEI, sentiment
    labels), so the v3w word-id path is checked on them too; also a two-question record mixing word ids and letters."""
    fever = {"type": "choice", "instructions": "Does the evidence support the claim?",
             "criteria": {"SUPPORTED": None, "REFUTED": None, "NOT_ENOUGH_INFO": "the evidence does not decide it"}}
    senti = {"type": "choice", "instructions": "What is the sentiment of the review?",
             "criteria": {"positive": None, "negative": None, "neutral": "neither positive nor negative"}}
    return [{"id": "synthetic:fever", "state": "Claim: the Eiffel Tower is in Rome. Evidence: it stands in Paris.",
             "questions": {"verdict": fever}, "labels": {"verdict": "REFUTED"}},
            {"id": "synthetic:sentiment", "state": "The food was cold and the staff ignored us.",
             "questions": {"sentiment": senti}, "labels": {"sentiment": "negative"}},
            {"id": "synthetic:mixed", "state": "A short review: great value, slow delivery.",
             "questions": {"sentiment": senti, "aspect": {"type": "choice", "instructions": "Which aspect is worst?",
                                                          "criteria": {"option_0": "price", "option_1": "delivery"}}},
             "labels": {"sentiment": "neutral", "aspect": "option_1"}}]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--n-small", type=int, default=120)
    ap.add_argument("--n-large", type=int, default=80)
    ap.add_argument("--n-multi", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--template", default="jrt-rwkv-g1-v2f",
                    choices=["jrt-rwkv-g1-v2", "jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"])
    ap.add_argument("--per-stratum", type=int, default=0,
                    help="> 0: pick_stratified (first N records of every family / k / id-style / dict-instruction "
                         "stratum) instead of the small / large / multi picker")
    ap.add_argument("--hist-vs", default=None, choices=["jrt-rwkv-g1-v2f"],
                    help="also build the rows with this template and report both gold-position / gold-target histograms")
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

    v3w = a.template == "jrt-rwkv-g1-v3w"
    if a.template in ("jrt-rwkv-g1-v2f", "jrt-rwkv-g1-v3w"):
        from decide.adapter import DEFAULT_LIMITS
        from decide.backends.hf_readout import RWKV7_G1K_TRAINED_CTX
        from decide.backends.rwkv_g1_filtered_codes import TokenCode
        from decide.backends.rwkv_g1_v3 import ANSWER_CTX, V3_CODES, G1V3Backend

        ro = G1V3Backend.__new__(G1V3Backend)  # the readout's own _questions, without loading a model
        ro.mode, ro.permute, ro.words, ro.calibrate, ro.reread = ("words" if v3w else "v2f"), False, v3w, False, False
        # the backend's default budget with the RWKV7-G1k plugin (load_rwkv7_hf max_context = RWKV7_G1K_TRAINED_CTX);
        # v2f never reads it (no code key)
        ro._first_tok, ro.budget = {}, min(RWKV7_G1K_TRAINED_CTX, DEFAULT_LIMITS["max_input_tokens"]) - 1
        ro._ctx_ids = readout_encode(ANSWER_CTX, True)
    else:
        ro = G1V2Backend.__new__(G1V2Backend)
    ro.plugin = SimpleNamespace(encode=readout_encode)
    ro.codes = (tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i) for i, e in enumerate(V3_CODES))
                if v3w else load_frozen_table())
    ro.code_table = [c.code for c in ro.codes]
    ro.code_token = {c.code: c.token_id for c in ro.codes}
    ro._full_memo = {}
    ctx = "User: Choose.\n\nAssistant: <think></think>\nANSWER:"  # the backend's own F4 proof, same tokenizer
    base = readout_encode(ctx, True)
    assert all(readout_encode(ctx + " " + c.code, True) == base + [c.token_id] for c in ro.codes)

    # (k > 255 never reaches a row: decide.schema.validate_legacy caps options at 255 = the code table size)
    records = (pick_stratified(a.records, a.per_stratum) if a.per_stratum > 0
               else pick_records(a.records, a.n_small, a.n_large, a.n_multi)) + [dict_instruction_record()] + word_id_records()
    n_rows, mism, by_k, refused = 0, [], {"<=16": 0, "17-255": 0, ">255": 0}, 0
    from collections import Counter

    fams, kinds, hist = Counter(), Counter(), {}

    def gold_hist(tpl: str, rows) -> None:  # rows: [(item, row)]; gold listing position (fifths for k > 4) + target
        h = hist.setdefault(tpl, {"position": {}, "gold_target": Counter()})
        for it, r in rows:
            k = r.n_options
            b = str(k) if k <= 4 else ("5-16" if k <= 16 else "17-255")
            pos = it.label if k <= 4 else (it.label * 5) // k
            h["position"].setdefault(b, Counter())[pos] += 1
            h["gold_target"][tok.decode(r.continuations[it.label])] += 1
    span_bad, empty_state, recollapsed = 0, 0, 0
    for rec in records:
        req = record_request(rec)
        try:
            items = items_from_records([rec], enc, layout="g1v2", template=a.template)
        except ControlTokenError:
            refused += 1
            continue
        qs = {q["qid"]: q for q in ro._questions(req)}
        gold_hist(a.template, [(it, it.row) for it in items])
        if a.hist_vs:
            gold_hist(a.hist_vs, [(it, it.row) for it in items_from_records([rec], enc, layout="g1v2",
                                                                            template=a.hist_vs)])
        fams[rec.get("family")] += 1
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
            if v3w:
                kinds["word_id" if q["words"] else ("code+key" if q["catalog_wanted"] else
                                                     "code" if q["coded"] else "fallback")] += 1
                kinds[f"catalog_chars={q['catalog_chars']}"] += q["catalog_wanted"]
                kinds["dict_instruction"] += not isinstance(req["questions"][qid].get("instructions"), (str, type(None)))
                want_n = q["catalog_chars"] if q["catalog_wanted"] else None
                if r.meta["words"] != q["words"] or r.meta["catalog_chars"] != want_n:
                    errs.append("v3w_words_or_catalog")
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
    hist_out = {t: {"position": {b: {str(p): c for p, c in sorted(v.items())} for b, v in sorted(h["position"].items())},
                    "gold_target_top": h["gold_target"].most_common(25), "gold_target_distinct": len(h["gold_target"])}
                for t, h in hist.items()}
    res = {"template": a.template, "records": a.records, "n_records": len(records), "n_rows": n_rows, "rows_by_k": by_k,
           "families": dict(sorted(fams.items(), key=str)), "row_kinds": dict(kinds), "gold_hist": hist_out,
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
