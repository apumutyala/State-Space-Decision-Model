"""Token streams for the pilot (wikitext, read row-group by row-group to keep laptop RAM low).

train : wikitext-103-raw-v1 train   (distillation tokens)
gate  : wikitext-103-raw-v1 validation (per-layer accept/rollback gate)
eval  : wikitext-2-raw-v1 test       (held-out perplexity; wt2 test == wt103 test, disjoint from wt103 train)
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np
import torch

DATA_ROOT = os.environ.get("SSDM_DATA_ROOT", r"C:\Users\apumu\research\CHLabs\ssdm-data")
WT = os.path.join(DATA_ROOT, "hf", "datasets--Salesforce--wikitext", "snapshots")
TOK_DIR = os.path.join(DATA_ROOT, "tokens")

SPLITS = {
    "train": ("wikitext-103-raw-v1", "train-*.parquet"),
    "gate": ("wikitext-103-raw-v1", "validation-*.parquet"),
    "eval": ("wikitext-2-raw-v1", "test-*.parquet"),
}


def _files(split):
    sub, pat = SPLITS[split]
    fs = sorted(glob.glob(os.path.join(WT, "*", sub, pat)))
    if not fs:
        raise FileNotFoundError(split)
    return fs


def sha256(path, nbytes=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def build(tok, split, max_tokens=None, doc_chars=20000):
    """Tokenise a split into a uint32 file. Text is grouped into ~doc_chars blocks so tokenisation is batched."""
    import pyarrow.parquet as pq

    os.makedirs(TOK_DIR, exist_ok=True)
    out = os.path.join(TOK_DIR, f"{split}.qwen35.u32")
    meta_p = out + ".json"
    if os.path.exists(meta_p):
        return out, json.load(open(meta_p))
    n = 0
    with open(out + ".tmp", "wb") as fo:
        for fp in _files(split):
            pf = pq.ParquetFile(fp)
            for rg in range(pf.num_row_groups):
                lines = pf.read_row_group(rg, columns=["text"]).column("text").to_pylist()
                buf, blocks = [], []
                size = 0
                for ln in lines:
                    buf.append(ln)
                    size += len(ln)
                    if size >= doc_chars:
                        blocks.append("".join(buf))
                        buf, size = [], 0
                if buf:
                    blocks.append("".join(buf))
                ids = tok(blocks, add_special_tokens=False)["input_ids"]
                for x in ids:
                    a = np.asarray(x, dtype=np.uint32)
                    fo.write(a.tobytes())
                    n += len(a)
                del lines, blocks, ids
                if max_tokens and n >= max_tokens:
                    break
            if max_tokens and n >= max_tokens:
                break
    os.replace(out + ".tmp", out)
    meta = {"split": split, "source_files": [os.path.basename(os.path.dirname(f)) + "/" + os.path.basename(f) for f in _files(split)],
            "source_sha256": {os.path.basename(f): sha256(f) for f in _files(split)},
            "n_tokens": n, "tokens_sha256": sha256(out), "tokenizer": tok.name_or_path, "max_tokens": max_tokens}
    json.dump(meta, open(meta_p, "w"), indent=1)
    return out, meta


def load_tokens(path):
    return np.memmap(path, dtype=np.uint32, mode="r")


class Batcher:
    """Random contiguous windows from a token memmap (seeded)."""

    def __init__(self, toks, seq_len, batch, seed, device="cuda", lo=0, hi=None):
        self.t, self.L, self.B = toks, seq_len, batch
        self.lo, self.hi = lo, (hi or len(toks)) - seq_len - 1
        self.rng = np.random.default_rng(seed)
        self.device = device

    def __next__(self):
        starts = self.rng.integers(self.lo, self.hi, size=self.B)
        x = np.stack([np.asarray(self.t[s:s + self.L], dtype=np.int64) for s in starts])
        return torch.from_numpy(x).to(self.device, non_blocking=True)

    def __iter__(self):
        return self


def fixed_windows(toks, seq_len, n, offset=0):
    """Deterministic, non-overlapping windows (for eval/gate)."""
    out = []
    for k in range(n):
        s = offset + k * seq_len
        if s + seq_len > len(toks):
            break
        out.append(np.asarray(toks[s:s + seq_len], dtype=np.int64))
    return torch.from_numpy(np.stack(out))
