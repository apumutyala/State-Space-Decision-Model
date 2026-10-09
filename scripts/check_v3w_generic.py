"""Checks of the generic v3w factory (decide.v3w_generic.make_v3w) on real DEV-subset requests (CPU, tokenizers only).

Per backbone (``--backbones``, keys of decide.v3w_generic.BACKBONES):
  table      code-table size, how many codes are also one token WITHOUT the leading space, the table is the ordered
             candidate list (contiguous prefix after single-token filtering)
  in_context for every rendered prompt: every code the question uses (and, on the first ``--full-table-prompts``
             prompts, every code of the table) is exactly one new token after the WHOLE prompt string, with and
             without the leading space; word-id first tokens likewise
  words      word-id coverage: questions in word mode / questions whose ids are all non-opaque / all questions
  train      train == readout: V3W.train_row(...).ids == V3W.questions(...)'s prompt ids, option order equal,
             continuations == the scored tokens, answer_pos == len(prompt) - 1
  template   (chat-template formats) the model's own template renders the v3w body verbatim between head and tail
For rwkv7-g1k-13.3b additionally (``legacy``):
  byte_identity  the factory's text and [0] + whole-string ids == today's G1V3Backend(mode="words")._questions
  table_rederived  build_code_table(RWKV encoder, ANSWER_CTX) == decide/data/rwkv_g1_v3_codes.json
  generic_core   the generic (non-legacy) code path with the RWKV format gives the same texts, ids and train rows

    set CUDA_VISIBLE_DEVICES= & python -X utf8 scripts/check_v3w_generic.py --rows <dev-rows.jsonl.gz> --n 500 \
        [--backbones rwkv7-g1k-13.3b qwen3.5-9b ...] [--out report.json]
"""
from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def dev_requests(rows_path: str, subset_path: str, n: int) -> list[tuple[str, dict]]:
    """The first ``n`` DEV rows (file order) whose run_id is in the pre-registered DEV subset, as requests."""
    ids = set(json.loads(Path(subset_path).read_text(encoding="utf-8"))["run_ids"])
    out = []
    with gzip.open(rows_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["_evaluation"]["run_id"] in ids:
                out.append((r["_evaluation"]["run_id"], {"state": r["state"], "questions": r["questions"]}))
                if len(out) >= n:
                    break
    return out


def legacy_backend(v):
    """Today's v3w readout's own _questions (G1V3Backend mode "words") on a model-free plugin, as
    scripts/check_g1v2_train_equivalence.py builds it."""
    from decide.backends.rwkv_g1_filtered_codes import TokenCode
    from decide.backends.rwkv_g1_v3 import ANSWER_CTX, V3_CODES, G1V3Backend

    ro = G1V3Backend.__new__(G1V3Backend)
    ro.mode, ro.permute, ro.words, ro.calibrate, ro.reread = "words", False, True, False, False
    ro._first_tok, ro.budget = {}, v.budget
    ro.plugin = SimpleNamespace(encode=lambda t, trusted=True: v.encode(t, trusted))
    ro._ctx_ids = ro.plugin.encode(ANSWER_CTX, True)
    ro.codes = tuple(TokenCode(code=e["code"], token_id=e["token_id"], source_ordinal=i) for i, e in enumerate(V3_CODES))
    ro.code_table = [c.code for c in ro.codes]
    ro.code_token = {c.code: c.token_id for c in ro.codes}
    ro._full_memo = {}
    return ro


def check_backbone(bid: str, reqs, root, full_table_prompts: int, legacy: bool) -> dict:
    from decide.backends.base import BackendUnsupported
    from decide.prompt_g1v3 import code_candidates, is_opaque_id
    from decide.schema import canonical_options
    from decide.untrusted import ControlTokenError
    from decide.v3w_generic import (RWKV_G1, V3W, build_code_table, check_template_passthrough, make_v3w)

    t0 = time.time()
    v = make_v3w(bid, root)
    cands = code_candidates()
    table = v.table
    res: dict = {"backbone": bid, "template_id": v.template_id, "identity_sha256": v.identity()["sha256"],
                 "format": {k: val for k, val in v.fmt.identity().items() if k in ("source", "no_think", "system_role",
                                                                                    "bos", "template_sha256")},
                 "head": v.fmt.head_text()[:400], "tail": v.fmt.tail, "budget": v.budget,
                 "table": {"size": len(table), "first": table[:3], "last": table[-3:],
                           "ordered_candidate_subsequence": table == [c for c in cands if c in set(table)],
                           "nospace_single_token": sum(e.get("nospace_token_id") is not None for e in v.codes)
                           if "nospace_token_id" in v.codes[0] else None}}
    if legacy:
        rebuilt = build_code_table(v.encode, v.fmt.answer_ctx)
        res["table"]["nospace_single_token"] = sum(e["nospace_token_id"] is not None for e in rebuilt)
        res["table_rederived"] = [(e["code"], e["token_id"]) for e in rebuilt] == [(e["code"], e["token_id"])
                                                                                  for e in v.codes]
        ro = legacy_backend(v)
        gv = V3W(v.backbone, dataclasses.replace(RWKV_G1), v.encode, v.codes, v.budget, v.template_id, v.control_ids)
        assert not gv.is_legacy_rwkv
    st = Counter()
    bad: list = []
    ctx_fail: list = []
    tok_dir = Path(root) / v.backbone.tokenizer_dir if root else None
    for n_req, (rid, req) in enumerate(reqs):
        try:
            qs = v.questions(req)
        except (BackendUnsupported, ControlTokenError) as e:
            st["refused_requests"] += 1
            bad.append({"id": rid, "refused": str(e)[:120]})
            continue
        st["requests"] += 1
        if legacy:
            lq = ro._questions(req)
            gq = gv.questions(req)
            for a, b, c in zip(qs, lq, gq):
                same = (a["text"] == b["text"] and a["prompt"] == b["prompt"] and a["options"] == b["options"]
                        and a.get("tokens") == b.get("tokens") and a["words"] == b["words"]
                        and a["catalog_chars"] == b["catalog_chars"] and a.get("conts") == b.get("conts"))
                st["byte_identity_rows"] += 1
                st["byte_identity_mismatch"] += not same
                st["generic_core_mismatch"] += not (c["text"] == a["text"] and c["prompt"] == a["prompt"]
                                                    and c.get("tokens") == a.get("tokens"))
                if not same:
                    bad.append({"id": rid, "qid": a["qid"], "kind": "byte_identity"})
            if len(lq) != len(qs):
                st["byte_identity_mismatch"] += 1
        for q in qs:
            st["questions"] += 1
            rq = req["questions"][q["qid"]]
            oids = [o for o, _ in canonical_options(rq)]
            nonopaque = len(oids) >= 2 and not any(is_opaque_id(o) or not o.strip() for o in oids)
            st["q_nonopaque_ids"] += nonopaque
            st["q_word_mode"] += q["words"]
            st["q_coded_letters"] += q["coded"] and not q["words"]
            st["q_fallback_multitoken"] += not q["coded"]
            st[f"catalog_chars={q['catalog_chars']}"] += q["catalog_wanted"]
            # in-context single-token proofs on the WHOLE prompt string
            body = q["prompt"][len(v.fmt.bos):]
            if q["coded"]:
                if q["words"]:
                    forms = [(" " + o, t) for o, t in zip(oids, q["tokens"])]
                else:
                    use = q["codes"] if n_req >= full_table_prompts else table
                    forms = [(" " + c, v.code_token[c]) for c in use]
                for s, t in forms:
                    ids = list(v.encode(q["text"] + s, True))
                    ok = ids[: len(body)] == body and (ids[len(body)] == t if len(ids) > len(body) else False)
                    if not q["words"]:
                        ok = ok and len(ids) == len(body) + 1
                    st["ctx_checks"] += 1
                    if not ok:
                        st["ctx_fail"] += 1
                        if len(ctx_fail) < 10:
                            ctx_fail.append({"id": rid, "qid": q["qid"], "form": s})
                    if not q["words"]:
                        ns = list(v.encode(q["text"] + s[1:], True))
                        st["ctx_nospace_checks"] += 1
                        st["ctx_nospace_single"] += ns[: len(body)] == body and len(ns) == len(body) + 1
            # train == readout
            try:
                row = v.train_row(req, q["qid"])
            except Exception as e:  # noqa: BLE001 - reported
                st["train_errors"] += 1
                bad.append({"id": rid, "qid": q["qid"], "train_error": f"{type(e).__name__}: {str(e)[:160]}"})
                continue
            errs = []
            if row.ids != q["prompt"]:
                errs.append("ids")
            if row.option_ids != q["options"]:
                errs.append("options")
            if row.answer_pos != len(q["prompt"]) - 1:
                errs.append("answer_pos")
            if q["coded"] and row.continuations != [[t] for t in q["tokens"]]:
                errs.append("target")
            if not q["coded"] and row.continuations != q["conts"]:
                errs.append("target_fallback")
            if row.meta["words"] != q["words"] or row.meta["catalog_chars"] != (q["catalog_chars"] if q["catalog_wanted"] else None):
                errs.append("words_or_catalog")
            st["train_rows"] += 1
            if legacy:
                grow = gv.train_row(req, q["qid"])
                if (grow.ids, grow.continuations, grow.state_span, grow.field_span, grow.option_spans) != (
                        row.ids, row.continuations, row.state_span, row.field_span, row.option_spans):
                    errs.append("generic_core_train_row")
            if errs:
                st["train_mismatch"] += 1
                if len(bad) < 30:
                    bad.append({"id": rid, "qid": q["qid"], "errors": errs})
            if tok_dir is not None and st["template_checks"] < 40 and v.fmt.source == "chat_template":
                st["template_checks"] += 1
                st["template_passthrough_fail"] += not check_template_passthrough(
                    v.fmt, tok_dir, q["text"][len(v.fmt.head_text()):len(q["text"]) - len(v.fmt.tail)])
    res["stats"] = dict(st)
    res["examples"] = bad[:15]
    res["ctx_fail_examples"] = ctx_fail
    res["seconds"] = round(time.time() - t0, 1)
    return res


def main() -> int:
    from decide.v3w_generic import BACKBONES, DEFAULT_TOKENIZERS

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--subset", default=str(Path(__file__).resolve().parents[1] / "jobs/common/dev_subset_m1.json"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--backbones", nargs="+", default=list(BACKBONES))
    ap.add_argument("--tokenizers", default=str(DEFAULT_TOKENIZERS))
    ap.add_argument("--full-table-prompts", type=int, default=3)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    reqs = dev_requests(a.rows, a.subset, a.n)
    out = {"n_requests": len(reqs), "backbones": []}
    ok = True
    for bid in a.backbones:
        r = check_backbone(bid, reqs, a.tokenizers, a.full_table_prompts, BACKBONES[bid].fmt == "rwkv_g1")
        s = r["stats"]
        ok &= not s.get("train_mismatch") and not s.get("train_errors") and not s.get("ctx_fail") \
            and not s.get("byte_identity_mismatch") and not s.get("generic_core_mismatch") \
            and not s.get("template_passthrough_fail") and r.get("table_rederived", True)
        out["backbones"].append(r)
        print(json.dumps({k: r[k] for k in ("backbone", "table", "stats", "seconds")}, ensure_ascii=True), flush=True)
        if a.out:
            Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    out["ok"] = ok
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
