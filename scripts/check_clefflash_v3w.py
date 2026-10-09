"""CPU check for arm-pair B of jobs/q9af_v2 mode ``clefflash`` (the LM-logit v3w readout of Clef-flash's backbone).

    python -X utf8 scripts/check_clefflash_v3w.py --rows <dev-rows.jsonl.gz> --clef-tok <dir with Clef-flash
        tokenizer.json + tokenizer_config.json> [--tokenizers <root>] [--out analysis/check_clefflash_v3w.json]

The readout is ``decide.v3w_generic.make_v3w("qwen3.5-9b")`` unchanged (PR #236: Qwen3.5-9B's own chat template, empty
think block, the generic code table). Clef-flash ships its own tokenizer.json (sha256 06b95093..., a different file
from Qwen3.5-9B's 5f9e4d49...; Clef-flash is a merged LoRA on Qwen3.5-9B: same vocab, a newer pre-tokenizer regex
that keeps combining marks with letters, and 7 extra audio/TTS added tokens). The job feeds the Qwen3.5-9B v3w ids
(the readout as merged, also the conversion's own tokenization). On every proxy600 row this script checks:
  * GATING (exit 1 otherwise): 255 codes; every scored token of every coded / word question is ONE token after the
    answer prefix and distinct within its question; no multi-token fallback is needed or it is reported;
  * REPORTED: how many questions Clef-flash's own tokenizer would tokenize differently (prompt ids / scored tokens), and
    whether its code table is the same.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def clef_v3w(v, clef_tok_dir: Path):
    """The same V3W (format, codes, budget, template id) over Clef-flash's tokenizer."""
    from decide.v3w_generic import V3W, HFEncoder, build_code_table

    enc = HFEncoder(clef_tok_dir)
    codes = build_code_table(enc, v.fmt.answer_ctx, len(v.codes), False, enc.control_ids)
    return V3W(v.backbone, v.fmt, enc, codes, v.budget, v.template_id, enc.control_ids)


def fork_volume(qs, kv_budget: int | None, limit: int) -> dict | None:
    """Exact prefix-fork positions / calls of decide.backends.v3w_generic_backend for one request (None = the adapter
    refuses it: a prompt + its answer position over ``limit``). ``kv_budget`` = the stock arm's KV fork budget (its
    attention layers' KV is copied per packed row); the all-GDN converted arm has none (fewer or equal calls)."""
    from decide.backends.rwkv_g1_v2 import PACK_ROWS, PACK_TOKENS
    from decide.backends.state_fork import common_prefix_len

    if any(len(q["prompt"]) + (1 if q["coded"] else max(map(len, q["conts"]))) > limit for q in qs):
        return None
    coded = [q for q in qs if q["coded"]]
    L = common_prefix_len([q["prompt"] for q in coded]) if coded else 0
    tails = sorted(((q, q["prompt"][L:]) for q in coded), key=lambda t: (-len(t[1]), t[0]["qid"]))
    packs, cur = [], []
    for t in tails:
        if cur and (len(cur) >= PACK_ROWS or len(cur[0][1]) * (len(cur) + 1) > PACK_TOKENS):
            packs.append(cur)
            cur = []
        cur.append(t)
    if cur:
        packs.append(cur)
    if kv_budget and L > 0:
        cap = max(1, kv_budget // L)
        packs = [pk[i:i + cap] for pk in packs for i in range(0, len(pk), cap)]
    calls = (1 if packs and L > 0 else 0) + len(packs)
    pos = L + sum(len(pk[0][1]) * len(pk) for pk in packs)  # padded positions per packed call
    tokens = L + sum(len(t) for pk in packs for _, t in pk)
    for q in qs:  # fallback questions (none on proxy600; counted as recompute per option)
        if not q["coded"]:
            calls += 1 + len(q["options"])
            tokens += len(q["prompt"]) + sum(len(c) for c in q["conts"])
            pos += len(q["prompt"]) * len(q["options"]) + sum(len(c) for c in q["conts"])
    return {"tokens": tokens, "calls": calls, "positions": pos}


def run(rows_path: Path, clef_tok: Path, tokenizers: Path | None, volume_out: Path | None = None) -> dict:
    from decide.backends.base import BackendUnsupported
    from decide.untrusted import ControlTokenError
    from decide.v3w_generic import make_v3w

    proxy = json.loads((ROOT / "jobs/common/dev_proxy600.json").read_text(encoding="utf-8"))
    want = set(proxy["run_ids"])
    v = make_v3w("qwen3.5-9b", tokenizers)
    c = clef_v3w(v, clef_tok)
    res = {"readout": v.template_id, "identity_sha256": v.identity()["sha256"],
           "format": {k: x for k, x in v.fmt.identity().items() if k in ("source", "no_think", "template_sha256")},
           "codes": {"qwen": len(v.codes), "clef": len(c.codes),
                     "same_codes_and_ids": [(e["code"], e["token_id"]) for e in v.codes]
                     == [(e["code"], e["token_id"]) for e in c.codes]},
           "control_ids_equal": v.control_ids == c.control_ids}
    from decide.adapter import DEFAULT_LIMITS
    from decide.backends.state_fork import KV_STATE_TOKEN_BUDGET

    st, bad = Counter(), []
    per_row: dict = {}
    with gzip.open(rows_path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rid = r["_evaluation"]["run_id"]
            if rid not in want:
                continue
            st["rows"] += 1
            req = {"state": r["state"], "questions": r["questions"]}
            out = []
            for vv in (v, c):
                try:
                    out.append(vv.questions(req))
                except (BackendUnsupported, ControlTokenError) as e:
                    out.append(f"refused: {type(e).__name__}")
            a, b = out
            if isinstance(a, str) or isinstance(b, str):
                st["refused_both" if a == b else "refused_mismatch"] += 1
                if a != b:
                    bad.append({"id": rid, "qwen": str(a)[:80], "clef": str(b)[:80]})
                continue
            vol = fork_volume(a, KV_STATE_TOKEN_BUDGET, DEFAULT_LIMITS["max_input_tokens"])
            st["rows_refused_over_limit"] += vol is None
            if vol is not None:
                per_row[rid] = {"fork": vol}
            for qa, qb in zip(a, b):
                st["questions"] += 1
                st["q_words"] += qa["words"]
                st["q_codes"] += qa["coded"] and not qa["words"]
                st["q_fallback_multitoken"] += not qa["coded"]
                st["q_over_budget"] += len(qa["prompt"]) > v.budget
                same = (qa["prompt"] == qb["prompt"] and qa.get("tokens") == qb.get("tokens")
                        and qa.get("conts") == qb.get("conts") and qa["options"] == qb["options"])
                st["q_ids_equal"] += same
                if qa["coded"]:
                    toks = qa["tokens"]
                    st["q_scored_tokens_single_and_distinct"] += (None not in toks and len(set(toks)) == len(toks))
                if not same:
                    bad.append({"id": rid, "qid": qa["qid"]})
            if len(a) != len(b):
                bad.append({"id": rid, "n_questions": [len(a), len(b)]})
    res["counts"] = dict(st)
    res["mismatches"] = bad[:50]
    res["n_mismatches"] = len(bad)
    res["control_ids_only_in_clef"] = sorted(c.control_ids - v.control_ids)
    tot = {"fork": {k: sum(r["fork"][k] for r in per_row.values()) for k in ("tokens", "calls", "positions")}}
    res["volume_fork_totals"] = tot["fork"]
    if volume_out is not None:
        Path(volume_out).write_text(json.dumps({
            "family": "q9af-probe-v3w", "template": v.template_id, "identity_sha256": v.identity()["sha256"],
            "rows": len(per_row), "refused": st["rows_refused_over_limit"], "totals": tot,
            "kv_fork_budget": KV_STATE_TOKEN_BUDGET, "limit": DEFAULT_LIMITS["max_input_tokens"],
            "source": "scripts/check_clefflash_v3w.py over jobs/common/dev_proxy600.json run_ids (stock arm: KV fork "
                      "budget applied; positions = padded packed positions)",
            "per_row": per_row}, indent=1) + "\n", encoding="utf-8", newline="\n")
    res["ok"] = (st["rows"] == len(want) and res["codes"]["qwen"] == 255
                 and st["q_scored_tokens_single_and_distinct"] == st["q_words"] + st["q_codes"])
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--clef-tok", required=True)
    ap.add_argument("--tokenizers", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--volume-out", default=None)
    a = ap.parse_args(argv)
    res = run(Path(a.rows), Path(a.clef_tok), Path(a.tokenizers) if a.tokenizers else None,
              Path(a.volume_out) if a.volume_out else None)
    txt = json.dumps(res, indent=1)
    if a.out:
        Path(a.out).write_text(txt + "\n", encoding="utf-8", newline="\n")
    print(txt)
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
