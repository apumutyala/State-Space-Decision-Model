"""Capability probes for the conversion pilot. All scoring is teacher-forced (no generation, no cache).

* ``mqar``  multi-query associative recall in natural tokens: N `key: value` pairs (key = single-token word,
            value = 3-digit number), then Q of the keys re-queried in random order; a query is correct iff all
            value digits are the argmax. Grid over N.
* ``niah``  single needle (5-digit passkey) in wikitext filler at several lengths x depths.
* ``ppl``   held-out perplexity (wikitext-2 test) at 1024 and 4096 context.
* ``mmlu``  0-shot MMLU sample (k per subject), letter scored by next-token logit over A/B/C/D.
"""
from __future__ import annotations

import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from convert.data import DATA_ROOT, fixed_windows

AC = dict(device_type="cuda", dtype=torch.bfloat16)


def _pad(ids, mult):
    # right-padding is invisible to earlier positions (causal model); bucketing avoids Triton re-autotuning per length
    n = (-len(ids)) % mult
    return ids + [0] * n


@torch.no_grad()
def _argmax_hits(model, ids: list[int], targets: list[tuple[int, int]], mult=256):
    x = torch.tensor([_pad(ids, mult)], device="cuda")
    with torch.autocast(**AC):
        h = model.hidden(x)[0]
    pos = torch.tensor([p - 1 for p, _ in targets], device="cuda")
    pred = model.lm_head(h[pos]).argmax(-1).tolist()
    return [int(pr == t) for pr, (_, t) in zip(pred, targets)]


# --------------------------------------------------------------------------- MQAR
_WORDS = None


def _key_words(tok, n=400):
    global _WORDS
    if _WORDS is None:
        cand = ("apple river stone cloud tiger maple orbit candle violin harbor desert falcon garden marble pepper "
                "rocket silver thunder window yellow anchor bridge castle dragon engine forest glacier hammer island "
                "jungle kettle ladder magnet needle ocean palace quartz rabbit saddle tunnel umbrella valley wagon "
                "zebra basket button camera cotton dinner doctor eagle finger flower guitar helmet insect jacket "
                "lemon lizard monkey napkin orange parrot pencil pillow planet pocket puzzle rocket salmon shadow "
                "spider sponge tomato turkey velvet walnut whistle wizard bottle carpet cherry circle copper cricket "
                "diamond dolphin feather fossil frog ginger goose honey iron jelly kitten lantern lobster meadow mirror "
                "muffin noodle oyster panda peanut pirate pumpkin robin rose saturn scarf shell shovel singer sofa "
                "statue sugar sunset tablet teapot tent tiger toast tower trumpet tulip turtle vase violet walrus "
                "wheat willow wolf yarn bamboo beacon blanket blossom bucket cactus canyon cedar chalk clover comet "
                "coral crystal dagger daisy denim echo ember fable fern flint fog galaxy gravel harp hazel iris ivory "
                "jade jasmine kayak kiwi lagoon lava lilac linen lotus mango marsh meteor mint moss nectar nickel "
                "olive onyx opal orchid otter pearl pebble pine plum polar prism quill raven reef ridge ruby sage "
                "sand sapphire satin shore silk slate snow spruce storm summit swan tide timber topaz torch trout "
                "tundra vapor vine wave willow zinc").split()
        seen, out = set(), []
        for w in cand:
            ids = tok(" " + w, add_special_tokens=False)["input_ids"]
            if len(ids) == 1 and w not in seen:
                seen.add(w)
                out.append(w)
        _WORDS = out
    return _WORDS


def mqar(model, tok, n_pairs=(8, 32, 128, 256), n_seq=20, n_q=16, seed=0):
    words = _key_words(tok)
    res = {}
    for N in n_pairs:
        N_eff = min(N, len(words))
        rng = np.random.default_rng(seed + N)
        hits = []
        for _ in range(n_seq):
            keys = list(rng.choice(words, size=N_eff, replace=False))
            vals = [f"{rng.integers(100, 1000)}" for _ in keys]
            ids = tok("Memorize these key-value pairs.\n", add_special_tokens=False)["input_ids"]
            for k_, v_ in zip(keys, vals):
                ids += tok(f" {k_}: {v_}\n", add_special_tokens=False)["input_ids"]
            ids += tok("Now recall the values.\n", add_special_tokens=False)["input_ids"]
            qs = rng.choice(N_eff, size=min(n_q, N_eff), replace=False)
            per_q = []
            for qi in qs:
                pre = tok(f" {keys[qi]}:", add_special_tokens=False)["input_ids"]
                vt = tok(f" {vals[qi]}", add_special_tokens=False)["input_ids"]
                ids += pre
                start = len(ids)
                ids += vt + tok("\n", add_special_tokens=False)["input_ids"]
                per_q.append([(start + j, t) for j, t in enumerate(vt)])
            flat = [t for q in per_q for t in q]
            h = _argmax_hits(model, ids, flat)
            c = 0
            for q in per_q:
                hits.append(int(all(h[c:c + len(q)])))
                c += len(q)
        res[f"N{N_eff}"] = {"acc": float(np.mean(hits)), "n": len(hits), "approx_len": len(ids)}
    res["mean_acc"] = float(np.mean([v["acc"] for k, v in res.items() if k.startswith("N")]))
    return res


# --------------------------------------------------------------------------- NIAH
def niah(model, tok, eval_toks, lengths=(512, 1024, 2048, 4096, 8192), depths=(0.1, 0.5, 0.9), n_per=8, seed=0):
    rng = np.random.default_rng(seed)
    res = {}
    for L in lengths:
        hits = []
        for d in depths:
            for r in range(n_per):
                key = str(rng.integers(10000, 100000))
                needle = tok(f" The secret passkey is {key}. Remember it: the secret passkey is {key}.",
                             add_special_tokens=False)["input_ids"]
                q = tok("\nQuestion: what is the secret passkey? Answer: the secret passkey is",
                        add_special_tokens=False)["input_ids"]
                ans = tok(f" {key}", add_special_tokens=False)["input_ids"]
                n_fill = L - len(needle) - len(q) - len(ans)
                off = int(rng.integers(0, len(eval_toks) - n_fill - 1))
                fill = np.asarray(eval_toks[off:off + n_fill], dtype=np.int64).tolist()
                at = int(d * n_fill)
                ids = fill[:at] + needle + fill[at:] + q
                start = len(ids)
                ids += ans
                h = _argmax_hits(model, ids, [(start + j, t) for j, t in enumerate(ans)])
                hits.append(int(all(h)))
        res[f"L{L}"] = {"acc": float(np.mean(hits)), "n": len(hits)}
    res["mean_acc"] = float(np.mean([v["acc"] for k, v in res.items() if k.startswith("L")]))
    return res


# --------------------------------------------------------------------------- PPL
@torch.no_grad()
def nll_windows(model, windows, bs=4):
    tot, n = 0.0, 0
    for s in range(0, len(windows), bs):
        x = windows[s:s + bs].cuda()
        with torch.autocast(**AC):
            h = model.hidden(x)
        for b in range(x.shape[0]):
            for c in range(0, x.shape[1] - 1, 512):  # chunked: full-vocab fp32 logits at 4k ctx would be ~4 GB
                e = min(c + 512, x.shape[1] - 1)
                lg = model.lm_head(h[b, c:e]).float()
                tot += F.cross_entropy(lg, x[b, c + 1:e + 1], reduction="sum").item()
            n += x.shape[1] - 1
    return tot / n


def ppl(model, eval_toks, cfgs=((1024, 64), (4096, 16))):
    out = {}
    for L, n in cfgs:
        w = fixed_windows(eval_toks, L, n)
        nll = nll_windows(model, w, bs=max(1, 4096 // L))
        out[f"ctx{L}"] = {"nll": nll, "ppl": math.exp(nll), "n_tokens": int(w.numel() - w.shape[0])}
    return out


# --------------------------------------------------------------------------- MMLU
def _mmlu_rows(k_per_subject=5, seed=0):
    import pyarrow.parquet as pq

    fp = sorted(glob.glob(os.path.join(DATA_ROOT, "hf", "datasets--cais--mmlu", "snapshots", "*", "all",
                                       "test-*.parquet")))[0]
    t = pq.read_table(fp).to_pylist()
    by = {}
    for r in t:
        by.setdefault(r["subject"], []).append(r)
    rng = np.random.default_rng(seed)
    rows = []
    for s in sorted(by):
        idx = rng.choice(len(by[s]), size=min(k_per_subject, len(by[s])), replace=False)
        rows += [by[s][i] for i in sorted(idx)]
    return rows, fp


@torch.no_grad()
def mmlu(model, tok, k_per_subject=5, seed=0):
    rows, fp = _mmlu_rows(k_per_subject, seed)
    letters = ["A", "B", "C", "D"]
    lid = [tok(" " + l, add_special_tokens=False)["input_ids"][-1] for l in letters]
    hits, preds = [], []
    for r in rows:
        subj = r["subject"].replace("_", " ")
        p = f"The following is a multiple choice question about {subj}.\n\n{r['question'].strip()}\n"
        for l, c in zip(letters, r["choices"]):
            p += f"{l}. {c}\n"
        p += "Answer:"
        ids = tok(p, add_special_tokens=False)["input_ids"]
        x = torch.tensor([_pad(ids, 128)], device="cuda")
        with torch.autocast(**AC):
            h = model.hidden(x)[0, len(ids) - 1]
        lg = model.lm_head(h).float()[lid]
        pr = int(lg.argmax())
        preds.append(pr)
        hits.append(int(pr == int(r["answer"])))
    return {"acc": float(np.mean(hits)), "n": len(hits), "k_per_subject": k_per_subject, "preds": preds,
            "source": os.path.basename(fp)}


def run_all(model, tok, eval_toks, cfg):
    p = cfg.get("probes", {})
    out = {}
    out["ppl"] = ppl(model, eval_toks, cfgs=tuple(tuple(x) for x in p.get("ppl", [[1024, 64], [4096, 16]])))
    out["niah"] = niah(model, tok, eval_toks, lengths=tuple(p.get("niah_lengths", [512, 1024, 2048, 4096, 8192])),
                       n_per=p.get("niah_n_per", 8))
    out["mqar"] = mqar(model, tok, n_pairs=tuple(p.get("mqar_pairs", [8, 32, 128, 256])), n_seq=p.get("mqar_n_seq", 20))
    out["mmlu"] = mmlu(model, tok, k_per_subject=p.get("mmlu_k", 5))
    return out
