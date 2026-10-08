"""Tree fork (prefix -> question -> option, each read once): exactness on real DEV prompts, tokens per row over the
full DEV length mix, the max-batch $ projection, and the training-side tokens per step. CPU only, $0, no weights.

    python -I -X utf8 analysis/tree_fork_tokens.py exact  [--n 200] [--json out.json]
    python -I -X utf8 analysis/tree_fork_tokens.py volume [--json out.json]
    python -I -X utf8 analysis/tree_fork_tokens.py train  [--steps 200] [--json out.json]

``exact``   a tiny fp32 RWKV-7 (the vendored 13.3B code; 2 layers, d 32, World vocab 65,536, random weights) + the fork
            head, on the D1 template (jrt-rwkv-g1-v1) prompts of ``--n`` DEV-subset requests tokenized by the real
            RWKV World tokenizer: head log-probs of ``prefix_mode="tree"`` vs ``"packed"`` (the current
            ``inference="fast"`` path) and ``"grouped"`` (the default path); max |diff|, argmax agreement.
``volume``  tokenizer only, every full-DEV request: tokens read per row by the packed path (``packed_plan``: every
            (question, option) tail re-reads the question suffix) and the tree (``tree_plan``), real tokens and
            positions incl. right padding (what the device computes), and calls; then the 2,000 common ok subset rows
            of analysis/throughput_maxbatch.py re-priced with the tree's positions at each arm's T1 per-token rate.
``train``   the D1 draw's first ``--steps`` optimiser steps (records in file order, one item per sorted label, the
            trainer's epoch-0 permutation for seed 0, D1's max_k / max_t filters; cohort / heldout exclusions are not
            re-applied, so this is the draw's distribution, not its exact rows), grouped as d1_fast does (row_batch 8,
            row_tokens 8,192, min_cost, cont_plan single): prefix tokens per step without / with ``share_prefix``.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # -I drops the script dir; the repo's own modules are trusted code

DATA = Path(r"C:\Users\apumu\research\CHLabs\ssdm-data")
DEV = DATA / "slices" / "dev-rows.jsonl.gz"
TRAIN = DATA / "m2" / "build" / "sample-150000-seed20261006" / "records.jsonl.gz"
TPL = "jrt-rwkv-g1-v1"  # D1's template


def _encoder():
    from heads.infer_opt import MemoEncoder
    from jobs.common.readout_cost import encoder

    return MemoEncoder(encoder("rwkv7"), max_entries=1 << 16)


def _dev_requests(ids: set[str] | None = None):
    with gzip.open(DEV, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rid = r["_evaluation"]["run_id"]
            if ids is None or rid in ids:
                yield rid, {"model": "t", "state": r["state"], "questions": r["questions"]}


def _rows(req, enc):
    from heads.layout import build_row

    return [build_row(req, q, enc, template=TPL, layout="ordered") for q in req["questions"]]


# ------------------------------------------------------------------------------------------------------------ exact
def _tiny_world_model(seed: int = 0):
    import torch

    from heads.backbone import BACKBONES, BackboneSpec, HiddenStateBackbone
    from heads.model import DecisionModel
    from third_party.rwkv7_hf.configuration_rwkv7 import Rwkv7Config
    from third_party.rwkv7_hf.modeling_rwkv7 import Rwkv7Model

    torch.manual_seed(seed)
    h = 32
    cfg = Rwkv7Config(vocab_size=65536, hidden_size=h, num_heads=h // 8, head_dim=8, num_hidden_layers=2,
                      intermediate_size=2 * h, a_low_rank_dim=8, decay_low_rank_dim=8, gate_low_rank_dim=8,
                      v_low_rank_dim=8)
    model = Rwkv7Model(cfg)
    with torch.no_grad():  # as heads.backbone.tiny_hf_backbone: randomise so the comparison is not vacuous
        for p in model.parameters():
            p.normal_(0.0, 0.1)
    full = BACKBONES["rwkv7-13.3b"]
    groups = tuple((g, rx, cnt * 2 // full.n_layers) for g, rx, cnt in full.lora_groups)
    spec = BackboneSpec(name="tiny-rwkv7-world", family="rwkv7", hf_id="random", params=0, n_layers=2, d_model=h,
                        lora_groups=groups, capture="hidden_states", attention_free=True)
    bb = HiddenStateBackbone(model, spec=spec, fork_mode="cache")
    return DecisionModel(bb, "fork", {}, pad_id=0, seed=seed).eval()


def exact(n: int) -> dict:
    import torch

    from analysis.throughput_maxbatch import load_rows
    from heads import prefix_features as PF

    torch.set_num_threads(4)
    common, _ = load_rows()
    # deterministic, length-unbiased pick: the n subset rows with the smallest sha256(run_id)
    pick = sorted(common, key=lambda r: hashlib.sha256(r.encode()).hexdigest())[:n]
    enc = _encoder()
    m = _tiny_world_model()
    assert PF.tree_supported(m.backbone)
    worst = {"packed": 0.0, "grouped": 0.0}
    agree = {"packed": 0, "grouped": 0}
    n_q = 0
    toks = []
    t0 = time.time()
    for rid, req in _dev_requests(set(pick)):
        try:
            rs = _rows(req, enc)
        except ValueError:
            continue
        with torch.no_grad():
            lp = {mode: m(rs, shared_prefix=True, prefix_mode=mode).log_probs.double()
                  for mode in ("tree", "packed", "grouped")}
        for i, r in enumerate(rs):
            k = r.n_options
            t = lp["tree"][i, :k]
            for ref in ("packed", "grouped"):
                o = lp[ref][i, :k]
                assert torch.isfinite(t).all() and torch.isfinite(o).all()
                worst[ref] = max(worst[ref], float((t - o).abs().max()))
                agree[ref] += int(int(t.argmax()) == int(o.argmax()))
            n_q += 1
        toks.append(sum(len(r.ids) for r in rs))
    return {"requests": len(toks), "questions": n_q, "max_abs_logprob_diff": worst, "argmax_identical": agree,
            "prompt_tokens_mean": float(np.mean(toks)), "prompt_tokens_max": int(max(toks)),
            "pass": all(v <= 1e-5 for v in worst.values()) and all(v == n_q for v in agree.values()),
            "seconds": time.time() - t0}


# ----------------------------------------------------------------------------------------------------------- volume
def volume() -> dict:
    from analysis.throughput_maxbatch import load_rows, run as tm_run
    from heads import prefix_features as PF

    enc = _encoder()
    per = {}
    refused = 0
    for rid, req in _dev_requests():
        try:
            rs = _rows(req, enc)
        except ValueError:
            refused += 1
            continue
        ids, conts = [r.ids for r in rs], [r.continuations for r in rs]
        per[rid] = {"packed": PF.packed_plan(ids, conts), "tree": PF.tree_plan(ids, conts),
                    "longest_question_row": max(len(x) for x in ids), "n_questions": len(rs),
                    "n_options": sum(r.n_options for r in rs)}
        if len(enc._memo) > 60000:
            enc._memo.clear()

    def stats(keys):
        out = {}
        for path in ("packed", "tree"):
            for f in ("tokens", "positions", "calls"):
                v = np.array([per[k][path][f] for k in keys], float)
                out[f"{path}_{f}"] = {"mean": float(v.mean()), "median": float(np.median(v)),
                                      "p95": float(np.percentile(v, 95)), "max": float(v.max())}
        lq = np.array([per[k]["longest_question_row"] for k in keys], float)
        out["longest_question_row_mean"] = float(lq.mean())
        tp = np.array([per[k]["tree"]["positions"] for k in keys], float)
        pp = np.array([per[k]["packed"]["positions"] for k in keys], float)
        out["tree_over_packed_positions"] = float(tp.sum() / pp.sum())
        out["packed_repeat_share_of_positions"] = float(1 - tp.sum() / pp.sum())
        return out

    res = {"template": TPL, "full_dev": {"requests": len(per), "refused": refused, **stats(list(per))}}
    common, _ = load_rows()
    sub = [k for k in common if k in per]
    res["subset_common"] = {"requests": len(sub), **stats(sub)}
    # re-price on the throughput model's rows, at each arm's T1 per-token rate (positions are what the device computes)
    tm = tm_run()
    proj = tm["projections"]
    n_old_model = tm["length_mix"]["rwkv7_g1v1"]["mean"]  # the stock readout's fork positions (the model's input)
    n_pk = res["subset_common"]["packed_positions"]["mean"]
    n_tree = res["subset_common"]["tree_positions"]["mean"]
    rate = tm["rate_usd_per_min"]

    def usd(s):
        return s / 60.0 * rate * 1000.0

    price = {"tokens_per_row": {"throughput_model_g1v1": n_old_model, "heads_packed": n_pk, "heads_tree": n_tree,
                                "g1v2_stock": tm["length_mix"]["rwkv7_v2"]["mean"]},
             "usd_per_1k_T1": {}}
    for name, arm in (("D1 + exact fixes, chunked_lean", "RWKV7 D1 + exact fixes, chunked_lean WKV"),
                      ("D1 + exact fixes + fla", "RWKV7 D1 + exact fixes + fla WKV")):
        s_tok = proj[arm]["per_token_us_T1"] / 1e6
        price["usd_per_1k_T1"][name] = {"model_as_published": proj[arm]["T1_usd_per_1k"],
                                        "packed (heads path)": usd(n_pk * s_tok), "tree": usd(n_tree * s_tok)}
    for c in ("Clef 27B (fast fla GDN)", "Clef-flash (fast fla GDN)"):
        price["usd_per_1k_T1"][c] = proj[c]["T1_usd_per_1k"]
    price["usd_per_1k_T1"]["stock g1-v2 + fla (reference)"] = proj["RWKV7 stock g1-v2, fla WKV"]["T1_usd_per_1k"]
    price["tokens_for_2x_vs_clef27b_at_equal_mfu"] = tm["rwkv_tokens_per_row_for_2x_at_equal_mfu"]["clef27b"]
    res["price"] = price
    return res


# ------------------------------------------------------------------------------------------------------------ train
def train(steps: int) -> dict:
    import torch

    from decide.backends.state_fork import common_prefix_len
    from heads.layout import build_row
    from train.guard import record_request
    from train.run_cell import keep_item
    from train.trainer import row_groups_min_cost, single_cont_fits

    n_lab = []
    with gzip.open(TRAIN, "rt", encoding="utf-8") as f:
        for line in f:
            n_lab.append(len(json.loads(line)["labels"]))
    starts = np.concatenate([[0], np.cumsum(n_lab)])
    n_items = int(starts[-1])
    order = torch.randperm(n_items, generator=torch.Generator().manual_seed(0 * 100_003 + 0)).tolist()
    need = order[: steps * 8 * 3]  # oversample: the max_k / max_t filters drop some
    rec_of = {it: int(np.searchsorted(starts, it, side="right") - 1) for it in need}
    want = set(rec_of.values())
    recs = {}
    with gzip.open(TRAIN, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in want:
                recs[i] = json.loads(line)
    enc = _encoder()
    items = []
    for it in need:
        ri = rec_of[it]
        rec = recs[ri]
        qid = sorted(rec["labels"])[it - int(starts[ri])]
        row = build_row(record_request(rec), qid, enc, template=TPL, layout="ordered")
        if keep_item(row, 16, 2048):
            items.append((ri, row))
        if len(items) >= steps * 8:
            break
    per_step = []
    for s in range(steps):
        rows = items[8 * s: 8 * s + 8]
        lens = [len(r.ids) for _, r in rows]
        groups = row_groups_min_cost(lens, 8, 8192, [r.n_options for _, r in rows], 0)
        single = len(groups) > 1 and single_cont_fits([r.n_options for _, r in rows], 0)
        g_real = sum(lens)
        g_pos = sum(len(g) * max(lens[i] for i in g) for g in groups)
        sh_real, sh_pos, extra = 0, 0, 0
        same_rec_pairs = sum(1 for a in range(8) for b in range(a + 1, 8) if rows[a][0] == rows[b][0])
        for g in groups:
            L = common_prefix_len([rows[i][1].ids for i in g]) if len(g) > 1 else 0
            sh_real += sum(lens[i] for i in g) - (len(g) - 1) * L
            sh_pos += L + len(g) * max(lens[i] - L for i in g)
            extra += int(L > 0)
        cont = sum(len(c) for _, r in rows for c in r.continuations)
        per_step.append({"groups": len(groups), "cont_single": single, "prefix_tokens": g_real,
                         "prefix_positions": g_pos, "shared_tokens": sh_real, "shared_positions": sh_pos,
                         "extra_calls": extra, "cont_tokens": cont, "same_record_pairs": same_rec_pairs})

    def mean(k):
        return float(np.mean([p[k] for p in per_step]))

    out = {k: mean(k) for k in per_step[0] if k != "cont_single"}
    out["steps"] = steps
    out["saved_positions_per_step"] = out["prefix_positions"] - out["shared_positions"]
    out["saved_share_of_step_positions"] = out["saved_positions_per_step"] / (out["prefix_positions"] + out["cont_tokens"])
    out["question_reread_per_option_in_training"] = 0  # fork_features_rows reads each row once, then K continuations
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("exact", "volume", "train"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    res = {"exact": lambda: exact(a.n), "volume": volume, "train": lambda: train(a.steps)}[a.what]()
    txt = json.dumps(res, indent=1)
    if a.json:
        Path(a.json).write_text(txt + "\n", encoding="utf-8")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
