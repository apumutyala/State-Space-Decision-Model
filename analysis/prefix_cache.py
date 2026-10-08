"""Cross-row prefix caching: how many prompt tokens are IDENTICAL across DEV rows (per family and globally), what a
cached preamble state / KV would save per row for RWKV7 (D1 template g1-v1, stock v2f / v3w) and Clef (its own
``encode_record`` layout), and the $/1k rows that follow at max batch. CPU only, $0, tokenizers only, no weights.

    python -I -X utf8 analysis/prefix_cache.py tokens --arm g1v1|v2f|v3w|clef --out <dir> [--limit N]
    python -I -X utf8 analysis/prefix_cache.py boundary --out <dir>
    python -I -X utf8 analysis/prefix_cache.py analyze --out <dir> [--json analysis/prefix_cache/result.json]

``tokens``    renders every full-DEV request with the arm's own prompt builder and tokenizer and stores, per request,
              the token sequence the model reads ONCE per request (RWKV: the common token prefix of the request's
              question prompts, which the tree / fork prefills once; Clef: its single joint sequence) plus the tokens
              actually computed per request (RWKV g1-v1: ``tree_plan`` positions; v2f / v3w: the backend's ``plan``
              total; Clef: the sequence length).
                g1v1  heads.layout.build_row(template jrt-rwkv-g1-v1, layout ordered), segment-wise G1k tokenization
                      (D1's RwkvSegmentEncoder; G1k tokenizer.json under ssdm-data/tokenizers)
                v2f / v3w  decide.prompt_g1v3.render_g1v3 at origin/claude/b-train-v3w-template @ 9c3e435 (no diff
                      vs r3-catalog @ 0290fa8 on the prompt / backend / table files), [0] + whole-string G1k
                      tokenization (F2, F5), v3w word ids by the backend's own first-token rule
                clef  baselines.clef_backend.ClefEncoder (clef_native ids; identical to ssdm ids on text without
                      control strings), Clef's own tokenizer.json (sha 06b95093..., the pinned one)
``boundary``  BPE / trie merge check at each candidate cache boundary: tok(preamble + rest) == tok(preamble) +
              tok(rest)? (whole-string arms), and where the token-space LCP ends relative to the text-space LCP.
``analyze``   longest common token prefix globally and per family (``family`` field), an ideal radix (trie) prefix
              cache over all rows, the share of each row's computed tokens, and the re-priced $/1k rows (2,000 common
              ok subset rows of analysis/throughput_maxbatch.py) with the cached tokens removed. Clef keeps the
              attention FLOPs of its suffix tokens over the cached prefix (2*qd*(n^2 - c^2) per full-attention layer).
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
DEV = DATA / "slices" / "dev-rows.jsonl.gz"
G1K_TOK = DATA / "tokenizers" / "RWKV--RWKV7-G1k-13.3B-20260930@ce3630b854cedbd673569f4600c6cf744bca9dba"
CLEF_TOK = DATA / "hf" / "clef-flash-tokenizer" / "17f0b0ad64efb65d273590632833508766b2aae6"
V3_SHA = "9c3e4350ccb41bec1385d58b9fe7002184f69626"  # origin/claude/b-train-v3w-template
CHUNK = 64  # fla chunk size (WKV7 / GDN): a chunk-aligned cache boundary keeps the prefix chunking of a one-pass read


def _dev():
    with gzip.open(DEV, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield r["_evaluation"]["run_id"], r["family"], {"model": "t", "state": r["state"], "questions": r["questions"]}


def _lcp(a, b) -> int:
    n = min(len(a), len(b))
    if n == 0:
        return 0
    a = np.asarray(a[:n]); b = np.asarray(b[:n])
    neq = np.nonzero(a != b)[0]
    return int(neq[0]) if len(neq) else n


def _g1k():
    sys.path.insert(1, str(REPO))
    from third_party.rwkv7_hf.tokenization_rwkv7 import Rwkv7Tokenizer

    return Rwkv7Tokenizer.from_pretrained(str(G1K_TOK))


# ------------------------------------------------------------------------------------------------------------ tokens
def _v3_tree(work: Path) -> Path:
    """The v3w branch's decide/ package, read from git (repo code, trusted) into ``work``."""
    dst = work / f"v3tree-{V3_SHA[:8]}"
    if not (dst / "decide" / "prompt_g1v3.py").exists():
        dst.mkdir(parents=True, exist_ok=True)
        arc = subprocess.run(["git", "-C", str(REPO), "archive", "--format=tar", V3_SHA, "decide"],
                             capture_output=True, check=True).stdout
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(arc)) as tf:
            tf.extractall(dst, filter="data")
    return dst


def tokens(arm: str, out: Path, limit: int | None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    rids, fams, prefixes, computed, nq, full_len = [], [], [], [], [], []
    t0 = time.time()
    if arm == "g1v1":
        sys.path.insert(0, str(REPO))
        from decide.backends.state_fork import common_prefix_len
        from heads import prefix_features as PF
        from heads.infer_opt import MemoEncoder
        from heads.layout import build_row
        from train.run_cell import RwkvSegmentEncoder

        tok = _g1k()
        enc = MemoEncoder(RwkvSegmentEncoder(tok, set(tok.all_special_ids)), max_entries=1 << 16)

        def one(req):
            rs = [build_row(req, q, enc, template="jrt-rwkv-g1-v1", layout="ordered") for q in req["questions"]]
            ids = [r.ids for r in rs]
            L = common_prefix_len(ids)
            plan = PF.tree_plan(ids, [r.continuations for r in rs])
            return ids[0][:L], plan["positions"], len(rs), max(len(x) for x in ids)
    elif arm in ("v2f", "v3w"):
        sys.path.insert(0, str(_v3_tree(out)))
        sys.path.insert(1, str(REPO))
        import decide.prompt_g1v3 as P3
        from decide.adapter import DEFAULT_LIMITS
        from decide.backends.rwkv_g1_filtered_codes import load_frozen_table
        from decide.backends.state_fork import common_prefix_len
        from decide.prompt import G1V2_TEMPLATE
        from decide.schema import canonical_options, canonical_question_ids

        assert Path(P3.__file__).resolve().is_relative_to(out.resolve()), P3.__file__
        tok = _g1k()
        enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
        table = [c.code for c in load_frozen_table()] if arm == "v2f" else [e["code"] for e in P3.load_v3_codes()[0]]
        budget = DEFAULT_LIMITS["max_input_tokens"] - 1
        ctx = enc("User: Choose.\n\nAssistant: <think></think>\nANSWER:")
        first = {}

        def first_token(o):
            if o not in first:
                ids = enc("User: Choose.\n\nAssistant: <think></think>\nANSWER:" + " " + o)
                first[o] = ids[len(ctx)] if len(ids) > len(ctx) and ids[:len(ctx)] == ctx and ids[len(ctx)] != 0 else None
            return first[o]

        def one(req):
            qs = req["questions"]
            words = (frozenset(k for k in canonical_question_ids(qs) if P3.word_ids_ok(qs[k], first_token))
                     if arm == "v3w" else frozenset())
            prompts, coded, conts = [], [], []
            for qid in canonical_question_ids(qs):
                ladder = (P3.CATALOG_CHARS_LADDER if arm == "v3w" and qid not in words and P3.wants_catalog(qs[qid], table)
                          else (None,))
                for n in ladder:
                    ids = [0] + enc(P3.render_g1v3(req, qid, table, n, False, True, words))
                    if len(ids) <= budget:
                        break
                prompts.append(ids)
                k = len(canonical_options(qs[qid]))
                c = qid in words or k <= len(table)
                coded.append(c)
                conts.append(0 if c else sum(len(enc(" " + o + G1V2_TEMPLATE.terminator)) for o, _ in canonical_options(qs[qid])))
            cp = [p for p, c in zip(prompts, coded) if c]
            L = common_prefix_len(cp) if cp else 0
            total = L + sum(len(p) - L for p in cp) + sum(len(p) + n for p, c, n in zip(prompts, coded, conts) if not c)
            base = cp[0] if cp else prompts[0]
            if not cp:  # every question uncoded: each prompt is its own call; the shared part is still a prefix
                L = common_prefix_len(prompts)
            return base[:L], total, len(prompts), max(len(p) for p in prompts)
    elif arm == "clef":
        sys.path.insert(0, str(REPO))
        from transformers import AutoTokenizer

        from baselines.clef_backend import ClefEncoder, load_jsm

        tok = AutoTokenizer.from_pretrained(str(CLEF_TOK))
        enc = ClefEncoder(tok, load_jsm(), "clef_native")

        def one(req):
            p = enc.parts(req)
            ids = p.prefix_ids + p.state_ids + p.schema_ids + p.suffix_ids
            return ids, len(ids), len(req["questions"]), len(ids)
    else:
        raise ValueError(arm)
    refused = 0
    for i, (rid, fam, req) in enumerate(_dev()):
        if limit and i >= limit:
            break
        try:
            pre, comp, q, fl = one(req)
        except Exception as exc:  # noqa: BLE001 - refused rows (control tokens, empty state) are counted, not priced
            refused += 1
            print("refused", rid, type(exc).__name__, str(exc)[:80], file=sys.stderr)
            continue
        rids.append(rid); fams.append(fam); prefixes.append(np.asarray(pre, np.int32)); computed.append(comp)
        nq.append(q); full_len.append(fl)
        if i % 1000 == 0:
            print(arm, i, round(time.time() - t0), "s", file=sys.stderr)
    off = np.concatenate([[0], np.cumsum([len(p) for p in prefixes])]).astype(np.int64)
    np.savez_compressed(out / f"tokens_{arm}.npz", ids=np.concatenate(prefixes), off=off,
                        computed=np.asarray(computed, np.int64), nq=np.asarray(nq), full_len=np.asarray(full_len),
                        rids=np.asarray(rids), fams=np.asarray(fams))
    print(json.dumps({"arm": arm, "requests": len(rids), "refused": refused, "seconds": round(time.time() - t0)}))


# ---------------------------------------------------------------------------------------------------------- boundary
def boundary(out: Path) -> dict:
    """At each candidate cache boundary of each arm's template, is the tokenization prefix-stable?"""
    sys.path.insert(0, str(REPO))
    tok = _g1k()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids  # noqa: E731
    from transformers import AutoTokenizer

    ctok = AutoTokenizer.from_pretrained(str(CLEF_TOK))
    cenc = lambda s: ctok(s, add_special_tokens=False).input_ids  # noqa: E731
    from decide.prompt import G1V2_SYSTEM_PROMPT, G1V2_TEMPLATE, SYSTEM_PROMPT, TEMPLATES
    from baselines.clef_backend import load_jsm

    jsm = load_jsm()
    g1v1_head = TEMPLATES["jrt-rwkv-g1-v1"].head.format(system=SYSTEM_PROMPT) + "SCHEMA FIELDS:\n"
    v2_head = G1V2_TEMPLATE.head.format(system=G1V2_SYSTEM_PROMPT) + "SCHEMA FIELDS:\n"
    clef_head = f"<|im_start|>system\n{jsm.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    cands = {
        # (encoder, preamble text, what follows it in every row's text)
        "v2/v3 head ... 'SCHEMA FIELDS:\\n'": (enc, v2_head, "FIELD 1\nID: "),
        "v2/v3 head ... 'FIELD 1\\nID: '": (enc, v2_head + "FIELD 1\nID: ", None),
        "g1v1 head ... 'SCHEMA FIELDS:\\n'": (enc, g1v1_head, "FIELD 1\nID: "),
        "clef system ... 'STATE:\\n'": (cenc, clef_head, None),
    }
    res = {}
    seen = defaultdict(int)
    stats = {k: {"rows": 0, "stable": 0, "examples": []} for k in cands}
    sys.path.insert(0, str(_v3_tree(out)))
    for rid, fam, req in _dev():
        if seen[fam] >= 20:  # 20 rows per family, every family
            continue
        seen[fam] += 1
        from decide.schema import canonical_question_ids

        q0 = canonical_question_ids(req["questions"])[0]
        full_v2 = __import__("decide.prompt", fromlist=["render_g1v2"]).render_g1v2(req, q0, [chr(65 + i) for i in range(26)])
        clef_text = clef_head + jsm.render(req["state"])
        for k, (e, pre, _) in cands.items():
            text = clef_text if k.startswith("clef") else full_v2 if k.startswith("v2") else None
            if text is None:  # g1v1 is tokenized per segment: the head is its own segment, stable by construction
                stats[k]["rows"] += 1; stats[k]["stable"] += 1
                continue
            assert text.startswith(pre), (k, text[:200])
            a, b = e(text), e(pre) + e(text[len(pre):])
            st = a == b
            stats[k]["rows"] += 1; stats[k]["stable"] += int(st)
            if not st and len(stats[k]["examples"]) < 3:
                j = _lcp(a, b)
                stats[k]["examples"].append({"rid": rid, "first_diff_token": j, "len_pre": len(e(pre)),
                                             "text_at_boundary": text[len(pre) - 10:len(pre) + 10]})
    res["prefix_stability"] = stats
    # continuation stability of each FAMILY preamble (the family's common token prefix): decode it, append many
    # different continuations, re-tokenize; backoff = how many trailing preamble tokens are NOT reproduced for some
    # continuation (0 = the cached state can be reused for ANY new row whose text starts with the preamble text).
    # Whole-string arms (v2f / v3w) are tokenized exactly so; g1v1 / clef tokenize per segment, where this is an
    # upper bound (a segment boundary inside the preamble is stable by construction).
    rng = np.random.default_rng(0)
    snippets = [chr(c) for c in range(32, 127)] + ["\n", "\t", "é", "中", "’", " \n", "0", "00", "{\""]
    states = []
    for rid, fam, req in _dev():
        s = jsm.render(req["state"])
        if len(s) > 40:
            j = int(rng.integers(0, len(s) - 30))
            states.append(s[j:j + 24])
        if len(states) >= 300:
            break
    snippets += states
    fam_res = {}
    for arm, e in (("g1v1", enc), ("v2f", enc), ("v3w", enc), ("clef", cenc)):
        f = out / f"tokens_{arm}.npz"
        if not f.exists():
            continue
        rows = _load(out, arm)
        st = prefix_stats(rows)
        first_of = {}
        for k, r in rows.items():
            first_of.setdefault(r["fam"], r["pre"])
        dec = (lambda ids: tok.decode(ids)) if arm != "clef" else (lambda ids: ctok.decode(ids))
        per = {}
        for fam, L in st["family_lcp"].items():
            if L <= 1:
                continue
            ids = [int(x) for x in first_of[fam][:L]]
            bos = 1 if arm in ("v2f", "v3w") and ids[0] == 0 else 0
            body = ids[bos:]
            txt = dec(body)
            rt = e(txt) == body
            worst = len(body)
            for c in snippets:
                worst = min(worst, _lcp(e(txt + c), body))
            per[fam] = {"lcp_tokens": L, "roundtrip_exact": rt, "backoff_tokens": len(body) - worst}
        bo = [v["backoff_tokens"] for v in per.values()]
        fam_res[arm] = {"families": len(per), "roundtrip_exact": sum(v["roundtrip_exact"] for v in per.values()),
                        "backoff_max": max(bo) if bo else None, "backoff_mean": float(np.mean(bo)) if bo else None,
                        "backoff_zero_families": sum(b == 0 for b in bo), "per_family": per}
    res["family_preamble_continuation_stability"] = fam_res
    res["n_continuations"] = len(snippets)
    res["preamble_tokens"] = {"g1v1 head": len(enc(g1v1_head)), "v2/v3 head (+[0])": 1 + len(enc(v2_head)),
                              "clef head": len(cenc(clef_head))}
    (out / "boundary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return res


# ----------------------------------------------------------------------------------------------------------- analyze
def _load(out: Path, arm: str) -> dict:
    z = np.load(out / f"tokens_{arm}.npz")
    ids, off = z["ids"], z["off"]
    rows = {}
    for i, rid in enumerate(z["rids"].tolist()):
        rows[rid] = {"fam": str(z["fams"][i]), "pre": ids[off[i]:off[i + 1]], "computed": int(z["computed"][i]),
                     "nq": int(z["nq"][i]), "full_len": int(z["full_len"][i])}
    return rows


def prefix_stats(rows: dict) -> dict:
    """Per row: fam_lcp (tokens common to EVERY row of its family), glob_lcp (to every row), radix_any (the longest
    prefix it shares with ANY other row), radix_amort (trie savings assigned to the later row in sorted order; the sum
    is the exact token saving of an ideal radix prefix cache over the set)."""
    keys = list(rows)
    pre = [rows[k]["pre"] for k in keys]
    glob = len(pre[0])
    for p in pre[1:]:
        glob = min(glob, _lcp(pre[0], p))
    fam_of = defaultdict(list)
    for i, k in enumerate(keys):
        fam_of[rows[k]["fam"]].append(i)
    fam_lcp = {}
    for f, idx in fam_of.items():
        L = len(pre[idx[0]])
        for j in idx[1:]:
            L = min(L, _lcp(pre[idx[0]], pre[j]))
        fam_lcp[f] = L if len(idx) > 1 else 0  # a family of one row has nothing to share
    order = sorted(range(len(keys)), key=lambda i: pre[i].astype(">u4").tobytes())
    adj = [0] + [_lcp(pre[order[t - 1]], pre[order[t]]) for t in range(1, len(order))]
    any_ = np.zeros(len(keys), int); amort = np.zeros(len(keys), int)
    for t, i in enumerate(order):
        amort[i] = adj[t]
        any_[i] = max(adj[t], adj[t + 1] if t + 1 < len(order) else 0)
    out = {}
    for i, k in enumerate(keys):
        L = len(pre[i])
        cap = max(L - 1, 0)  # the last prefix token is always computed (its features / state feed the fork or head)
        out[k] = {"fam": rows[k]["fam"], "L": L, "computed": rows[k]["computed"],
                  "glob": min(glob, cap), "fam_lcp": min(fam_lcp[rows[k]["fam"]], cap),
                  "radix_any": min(int(any_[i]), cap), "radix_amort": min(int(amort[i]), cap)}
        out[k]["fam_lcp_chunk"] = (out[k]["fam_lcp"] // CHUNK) * CHUNK
    return {"global_lcp": glob, "family_lcp": fam_lcp, "per_row": out}


def _summ(per: dict, keys: list[str]) -> dict:
    comp = np.array([per[k]["computed"] for k in keys], float)
    L = np.array([per[k]["L"] for k in keys], float)
    s = {"rows": len(keys), "computed_mean": float(comp.mean()), "prefix_mean": float(L.mean())}
    for m in ("glob", "fam_lcp", "fam_lcp_chunk", "radix_any", "radix_amort"):
        v = np.array([per[k][m] for k in keys], float)
        s[m] = {"mean_tokens": float(v.mean()), "median_tokens": float(np.median(v)),
                "share_of_computed": float(v.sum() / comp.sum()), "share_of_prefix": float(v.sum() / L.sum()),
                "mean_row_share": float(np.mean(v / comp))}
    return s


def _family_table(per: dict, keys: list[str], fam_lcp: dict) -> list[dict]:
    by = defaultdict(list)
    for k in keys:
        by[per[k]["fam"]].append(k)
    tab = []
    for f, ks in sorted(by.items(), key=lambda t: -len(t[1])):
        comp = np.array([per[k]["computed"] for k in ks], float)
        tab.append({"family": f, "rows": len(ks), "family_lcp_tokens": fam_lcp[f],
                    "computed_mean": float(comp.mean()),
                    "fam_lcp_share": float(np.sum([per[k]["fam_lcp"] for k in ks]) / comp.sum()),
                    "radix_amort_mean": float(np.mean([per[k]["radix_amort"] for k in ks])),
                    "radix_amort_share": float(np.sum([per[k]["radix_amort"] for k in ks]) / comp.sum())})
    return tab


def analyze(out: Path, json_path: Path | None) -> dict:
    sys.path.insert(0, str(REPO))
    from analysis.throughput_maxbatch import CLEF, PEAK_BF16, clef_flops, rwkv_flops, run as tm_run

    arms = [a for a in ("g1v1", "v2f", "v3w", "clef") if (out / f"tokens_{a}.npz").exists()]
    rows = {a: _load(out, a) for a in arms}
    tm = tm_run()
    # the 2,000 common ok rows of the throughput model (same ids as its $ table), and the 2,008-row subset
    from analysis.throughput_maxbatch import load_rows

    common, data = load_rows()
    subset = sorted(data["d1_child"])
    n_common_all = len(common)
    common = [k for k in common if all(k in rows[a] for a in arms)]  # refused / missing rows drop out (reported)
    res = {"arms": {}, "chunk": CHUNK, "common_rows_priced": len(common), "common_rows_throughput_model": n_common_all}
    stats = {}
    for a in arms:
        st = prefix_stats(rows[a])
        stats[a] = st
        per = st["per_row"]
        full = list(per)
        sub = [k for k in subset if k in per]
        com = [k for k in common if k in per]
        res["arms"][a] = {"global_lcp_tokens": st["global_lcp"], "full_dev": _summ(per, full),
                          "subset_2008": _summ(per, sub), "common_2000": _summ(per, com),
                          "families_full_dev": _family_table(per, full, st["family_lcp"]),
                          "families_subset": _family_table(per, sub, st["family_lcp"])}
    # sanity: our token counts vs the landed arms' reported model tokens on the common rows
    chk = {}
    for a, arm in (("clef", "clef27b_fast"), ("v3w", "rwkv7_stock_v3w")):
        if a in rows:
            ours = np.array([rows[a][k]["computed"] for k in common if k in rows[a]], float)
            theirs = np.array([data[arm][k]["model_tokens"] for k in common if k in rows[a]], float)
            chk[a] = {"ours_mean": float(ours.mean()), "landed_mean": float(theirs.mean()),
                      "exact_match_rows": int((ours == theirs).sum()), "rows": len(ours)}
    if "g1v1" in rows:
        ours = np.array([rows["g1v1"][k]["computed"] for k in common], float)
        chk["g1v1_tree_positions_mean"] = float(ours.mean())  # analysis/tree_fork/volume.json: 2,482.3
    res["token_count_check"] = chk

    # ---------------------------------------------------------------- re-pricing on the common 2,000 rows (T1 + roof)
    proj = tm["projections"]
    rate = tm["rate_usd_per_min"]
    usd = lambda s: s / 60.0 * rate * 1000.0  # noqa: E731
    clef_n = {c: np.array([data[arm][k]["model_tokens"] for k in common], float)
              for c, arm in (("clef27b", "clef27b_fast"), ("flash", "flash_fast"))}
    price = {}
    for c, name in (("clef27b", "Clef 27B (fast fla GDN)"), ("flash", "Clef-flash (fast fla GDN)")):
        n = clef_n[c]
        cfg = CLEF[c]
        out_c = {"no_cache": {"T1_usd_per_1k": proj[name]["T1_usd_per_1k"], "tokens": float(n.mean())}}
        if "clef" in stats:
            per = stats["clef"]["per_row"]
            for m in ("glob", "fam_lcp", "radix_amort"):
                cc = np.array([per[k][m] for k in common], float)
                L = cfg["num_hidden_layers"]; n_full = L // cfg["full_attention_interval"]
                qd = cfg["num_attention_heads"] * cfg["head_dim"]
                f0 = clef_flops(cfg, n)
                f1 = f0 - (clef_flops(cfg, cc) - 0.0)  # linear terms on c tokens + 2*qd*c^2 (the cached block's
                # own causal attention); the suffix's attention over the cached prefix stays in f1
                ratio = f1.sum() / f0.sum()
                out_c[m] = {"cached_tokens_mean": float(cc.mean()), "flops_ratio": float(ratio),
                            "T1_usd_per_1k": proj[name]["T1_usd_per_1k"] * float(ratio),
                            "flops_per_row_T": float(f1.mean() / 1e12),
                            "attn_over_cached_prefix_T": float((2.0 * n_full * qd * cc * (n - cc) * 2).mean() / 1e12)}
        price[name] = out_c
    rw = {}
    arm_rate = {"g1v1": ("RWKV7 D1 + exact fixes, chunked_lean WKV", "RWKV7 D1 + exact fixes + fla WKV"),
                "v2f": ("RWKV7 stock g1-v2, chunked WKV", "RWKV7 stock g1-v2, fla WKV"),
                "v3w": ("RWKV7 stock g1-v2, chunked WKV", "RWKV7 stock g1-v2, fla WKV")}
    for a in ("g1v1", "v2f", "v3w"):
        if a not in stats:
            continue
        per = stats[a]["per_row"]
        comp = np.array([per[k]["computed"] for k in common], float)
        r = {}
        for kern, pname in zip(("chunked", "fla"), arm_rate[a]):
            s_tok = proj[pname]["per_token_us_T1"] / 1e6
            r[kern] = {"no_cache": usd(comp.mean() * s_tok)}
            for m in ("glob", "fam_lcp", "fam_lcp_chunk", "radix_amort"):
                cc = np.array([per[k][m] for k in common], float)
                r[kern][m] = usd((comp - cc).mean() * s_tok)
        r["tokens_mean"] = {"no_cache": float(comp.mean()),
                            **{m: float((comp - np.array([per[k][m] for k in common], float)).mean())
                               for m in ("glob", "fam_lcp", "fam_lcp_chunk", "radix_amort")}}
        r["flops_per_row_T"] = {m: float(rwkv_flops(v) / 1e12) for m, v in r["tokens_mean"].items()}
        rw[a] = r
    price["rwkv"] = rw
    # 2x vs Clef 27B: under each cache policy (both sides cached the same way), what RWKV needs
    need = {}
    c27 = price["Clef 27B (fast fla GDN)"]
    for a in rw:
        for m in ("no_cache", "fam_lcp", "radix_amort"):
            cm = "no_cache" if m == "no_cache" else m
            clef_usd = c27[cm]["T1_usd_per_1k"] if cm in c27 else c27["no_cache"]["T1_usd_per_1k"]
            clef_s = clef_usd / 1000.0 / rate * 60.0  # s per row at Clef's T1 (its measured 38% large-call MFU)
            rw_tok = rw[a]["tokens_mean"][m]
            rw_fl = rwkv_flops(rw_tok)
            target_s = clef_s / 2.0
            clef_fl = (c27[cm]["flops_per_row_T"] * 1e12 if "flops_per_row_T" in c27.get(cm, {})
                       else float(clef_flops(CLEF["clef27b"], clef_n["clef27b"]).mean()))
            need[f"{a}/{m}"] = {"clef27b_T1_usd": clef_usd, "rwkv_fla_T1_usd": rw[a]["fla"][m],
                                "ratio_rwkv_over_clef_T1_fla": rw[a]["fla"][m] / clef_usd,
                                "rwkv_tokens": rw_tok, "rwkv_mfu_needed_for_2x": rw_fl / target_s / PEAK_BF16,
                                "rwkv_fla_T1_mfu": proj[arm_rate[a][1]]["T1_mfu"],
                                "clef_T1_mfu": proj["Clef 27B (fast fla GDN)"]["T1_mfu"],
                                "equal_mfu_ratio": rw_fl / clef_fl}
    price["two_x_vs_clef27b"] = need
    res["price"] = price
    res["usd_rate_per_min"] = rate
    txt = json.dumps(res, indent=1)
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(txt + "\n", encoding="utf-8")
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("tokens", "boundary", "analyze"))
    ap.add_argument("--arm")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    out = Path(a.out)
    if a.what == "tokens":
        tokens(a.arm, out, a.limit)
    elif a.what == "boundary":
        boundary(out)
    else:
        res = analyze(out, Path(a.json) if a.json else None)
        print(json.dumps({k: v for k, v in res.items() if k != "arms"}, indent=1))
        for arm, v in res["arms"].items():
            print(arm, "global", v["global_lcp_tokens"], json.dumps(v["common_2000"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
