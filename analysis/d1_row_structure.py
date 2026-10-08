# Usage: python -I -X utf8 analysis/d1_row_structure.py <repo> <rwkv7 tokenizer dir> <dev-rows.jsonl.gz> <units-child-index.json> <parent units-index.json> <out.json>
"""Per-row structure of the D1 child's scoring work on the 2,000 ok subset rows (CPU, tokenizer only):
backbone calls (grouped vs packed), tokens, tokenisation volume; then a least-squares attribution of child wall."""
import gzip, json, sys, time
import numpy as np
sys.path.insert(0, sys.argv[1])
from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer
from train.run_cell import RwkvSegmentEncoder
from heads.layout import build_row
from heads.infer_opt import MemoEncoder
from decide.backends.state_fork import common_prefix_len

tok = Rwkv7Tokenizer.from_pretrained(sys.argv[2]); enc = RwkvSegmentEncoder(tok, set(tok.all_special_ids))
child = json.load(open(sys.argv[4]))["index_results"]; parent = json.load(open(sys.argv[5]))["index_results"]
rows = {r["_evaluation"]["run_id"]: r for r in (json.loads(l) for l in gzip.open(sys.argv[3], "rt", encoding="utf-8"))
        if r["_evaluation"]["run_id"] in child}
out = []
t_all = time.perf_counter()
for rid, r in rows.items():
    c, p = child[rid], parent.get(rid)
    if c["status"] != "ok" or not p or p["status"] != "ok":
        continue
    req = {"model": "x", "state": r["state"], "questions": r["questions"]}
    t0 = time.perf_counter()
    rr = [build_row(req, q, enc, template="jrt-rwkv-g1-v1", layout="ordered") for q in req["questions"]]
    tok_s = time.perf_counter() - t0
    memo = MemoEncoder(enc)
    t0 = time.perf_counter()
    for _ in range(2):  # plan + score through the memo
        [build_row(req, q, memo, template="jrt-rwkv-g1-v1", layout="ordered") for q in req["questions"]]
    memo_s = time.perf_counter() - t0
    ids = [x.ids for x in rr]
    L = common_prefix_len(ids)
    suf = [len(x) - L for x in ids]
    G = len(set(suf))
    Q = len(rr)
    grouped_calls = 1 + G + Q  # prefix + one per distinct suffix length + one fork call per question
    tails = sorted((s + len(cn) for x, s in zip(rr, suf) for cn in x.continuations), reverse=True)
    packs, cur = 0, []
    for t in tails:
        if cur and (len(cur) >= 64 or cur[0] * (len(cur) + 1) > 8192):
            packs += 1; cur = []
        cur.append(t)
    packs += bool(cur)
    cont_tok = sum(len(cn) for x in rr for cn in x.continuations)
    out.append(dict(rid=rid, Q=Q, L=L, suf=sum(suf), G=G, grouped=grouped_calls, packed=1 + packs, cont=cont_tok,
                    tail_tok=sum(tails), tok_s=tok_s, memo_s=memo_s, child_ms=c["total_wall_ms"], parent_ms=p["total_wall_ms"]))
print("rows", len(out), "elapsed", round(time.perf_counter() - t_all), "s", file=sys.stderr)
json.dump(out, open(sys.argv[6], "w"))
