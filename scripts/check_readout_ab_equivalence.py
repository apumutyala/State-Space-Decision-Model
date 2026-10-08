"""Readout-A/B equivalence (CPU, tokenizer only, no model): the trained cell's layout-g1v2 rows under an EVAL readout
(heads.layout G1_EVAL_TEMPLATES: jrt-rwkv-g1-v3p / -v3w / -v2fp) read exactly what the stock R3 readout reads.

For every (DEV subset row, question):
  layout side  = heads.layout.build_row(request, qid, train.run_cell.parent_encoder("rwkv7", tokdir), template, "g1v2")
  readout side = decide.backends.rwkv_g1_v3.G1V3Backend._questions (mode permute / words; v2fp = mode v2f with the
                 permutation flag on, the exact v2fp definition) on a model-free plugin with the readout's encode body
and it asserts
  ids      : row.ids == the readout's prompt ids ([0] + whole-string tokens; incl. the code-key label ladder rung)
  options  : row.option_ids == the readout's canonical option order
  answer   : row.continuations == [[the readout's scored token of option j]] for every coded / word-id question (the
             presented code under a permutation, the id's first answer token under v3w), else the multi-token fallback
  slot     : row.answer_pos == len(prompt) - 1 (the position the readout reads)

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_readout_ab_equivalence.py --rows <index-rows.jsonl.gz> \
        --tokenizer-dir <dir with the RWKV7-G1k tokenizer.json> --template jrt-rwkv-g1-v3w [--per-bench 4]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
MODES = {"jrt-rwkv-g1-v3p": ("permute", True), "jrt-rwkv-g1-v3w": ("words", False), "jrt-rwkv-g1-v2fp": ("v2f", True)}


def stock_readout(template: str, readout_encode):
    """G1V3Backend's own prompt/token code (``_questions``) without a model, as the stock family builds it."""
    from decide.backends.rwkv_g1_filtered_codes import TokenCode, load_frozen_table
    from decide.backends.rwkv_g1_v3 import ANSWER_CTX, V3_CODES, V3_TABLE, G1V3Backend

    mode, permute = MODES[template]
    ro = G1V3Backend.__new__(G1V3Backend)
    ro.mode, ro.permute, ro.words, ro.calibrate, ro.reread = mode, permute, mode == "words", False, False
    ro._first_tok, ro.budget = {}, 16_384 - 1  # min(25,600 trained ctx, adapter default 16,384) - 1, as __init__
    ro.plugin = SimpleNamespace(encode=readout_encode)
    if mode == "v2f":
        ro.codes = load_frozen_table()
        ro.code_table = [c.code for c in ro.codes]
    else:
        ro.codes = tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i) for i, e in enumerate(V3_CODES))
        ro.code_table = list(V3_TABLE)
    ro.code_token = {c.code: c.token_id for c in ro.codes}
    ro._full_memo = {}
    ro._ctx_ids = base = readout_encode(ANSWER_CTX, True)
    assert all(readout_encode(ANSWER_CTX + " " + c.code, True) == base + [c.token_id] for c in ro.codes)
    return ro


def pick(rows_path: str, per_bench: int) -> list[dict]:
    """The first ``per_bench`` rows of every benchmark (catalog_id) in file order: every family's prompt shape."""
    out, n = [], collections.Counter()
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = r["_evaluation"]["catalog_id"]
            if n[c] < per_bench:
                n[c] += 1
                out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--tokenizer-dir", required=True)
    ap.add_argument("--template", required=True, choices=sorted(MODES))
    ap.add_argument("--per-bench", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    from decide.backends.base import BackendUnsupported
    from decide.untrusted import ControlTokenError
    from heads.layout import build_row
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
    from train.run_cell import parent_encoder

    enc = parent_encoder("rwkv7", a.tokenizer_dir)
    tok = Rwkv7Tokenizer.from_pretrained(a.tokenizer_dir)
    special = frozenset(int(i) for i in tok.all_special_ids) | {0}

    def readout_encode(text: str, trusted: bool = True) -> list[int]:  # decide.backends.hf_readout.load_rwkv7_hf
        ids = tok(text, add_special_tokens=False).input_ids
        if not trusted and special.intersection(ids):
            raise ControlTokenError("untrusted text encodes to an RWKV special token (<|endoftext|>)")
        return ids

    ro = stock_readout(a.template, readout_encode)
    rows = pick(a.rows, a.per_bench)
    stats = collections.Counter()
    mism = []
    for r in rows:
        req = {"model": "ssdm-m2", "state": r["state"], "questions": r["questions"]}
        try:
            qs = {q["qid"]: q for q in ro._questions(req)}
        except (ControlTokenError, BackendUnsupported):
            stats["refused_by_readout"] += 1
            continue
        for qid, q in qs.items():
            try:
                row = build_row(req, qid, enc, template=a.template, layout="g1v2")
            except ValueError as e:
                mism.append({"run_id": r["_evaluation"]["run_id"], "qid": qid, "errors": [f"build_row: {e}"[:200]]})
                continue
            stats["rows"] += 1
            stats["word_id_rows"] += bool(q.get("words"))
            stats["catalog_rows"] += q.get("catalog_chars") is not None
            stats["long_prompt_rows(>16383)"] += len(q["prompt"]) > 16_383
            errs = []
            if row.ids != q["prompt"]:
                errs.append("ids")
            if row.option_ids != q["options"]:
                errs.append("options")
            if row.answer_pos != len(q["prompt"]) - 1 or row.ids[0] != 0:
                errs.append("answer_pos")
            if q["coded"]:
                if row.continuations != [[t] for t in q["tokens"]]:
                    errs.append("answer_tokens")
            elif row.continuations != q["conts"]:
                errs.append("fallback_conts")
            if q.get("catalog_wanted") and row.meta.get("catalog_chars") != q["catalog_chars"]:
                errs.append("catalog_rung")
            if errs:
                mism.append({"run_id": r["_evaluation"]["run_id"], "qid": qid, "k": row.n_options, "errors": errs})
    res = {"template": a.template, "rows_file": a.rows, "n_dev_rows": len(rows), **stats, "mismatches": len(mism),
           "mismatch_examples": mism[:10]}
    text = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if not mism and stats["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
