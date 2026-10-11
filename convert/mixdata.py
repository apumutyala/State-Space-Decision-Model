"""Generic distillation text for Q9-AF v2 (jobs/q9af_v2): a pinned public mixture, DEV-decontaminated, never re-used.

    python -m convert.mixdata build --config configs/q9af_v2_h100.json --dev-rows <dev-rows.jsonl.gz> --out <dir>

Why (full8 diagnosis, jobs/q9af_v2/JOB_SPEC.md): job-2xe5w distilled on 40M wikitext-103 tokens only, re-read ~4.7
times over stage 1 + 8 stage-2 steps, and gated on wikitext validation, so ppl stayed within 4 % while full MMLU fell
12.2 pts and the Index 26 pts. v2 distils on a broad, chat-formatted mixture the teacher (Qwen3.5-9B, open weights) is
actually used on, with every token seen at most once.

Sources (all public, generic, pinned to an immutable HF revision in ``scripts/sf_fetch.py`` stage ``q9af_v2``; none of
them is a benchmark, an Index row, cohort v3, a TEST or held-out row, or targets a frozen family):
  web   HuggingFaceFW/fineweb-edu  sample/10BT/000_00000.parquet     (educational web text)
  chat  HuggingFaceH4/ultrachat_200k train_sft-0000{0,1}-of-00003      (multi-turn assistant dialogues), rendered in the
        Qwen3.5 chat format the readout uses (``<|im_start|>role ... <|im_end|>``, empty think block before every
        assistant turn, as ``decide.prompt`` jrt-chatml-v1)
  math  HuggingFaceTB/finemath     finemath-4plus/train-00000-of-00064 (math web pages)
Held-out (the end-to-end KL gate): the LAST row group of the web and math files (training reads row groups from the
front and stops at its quota; a quota that would reach the held-out row group is refused) and ultrachat test_sft.

Decontamination (pre-registered, reported): every document sharing ANY token 13-gram with any DEV row (state +
canonical questions JSON, ``slices/dev-rows.jsonl.gz``, all 12,104 rows: every Index benchmark incl. the frozen
families' DEV rows) is dropped. The DEV rows are used only to build this n-gram filter, never as text.

Output: ``train.u32`` (one stream: sources concatenated, docs separated by <|endoftext|>; batches draw permuted
non-overlapping windows, so each source enters every batch at its token share), ``heldout_<src>.u32`` and
``mix.json`` (counts, drop counts, file sha256s, token sha256s).
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

NGRAM = 13
_P = np.uint64(1000003)
SOURCES = ("web", "chat", "math")
#: env var (scripts/sf_fetch.py paths.env) -> snapshot dir of each source
SOURCE_ENV = {"web": "FINEWEB_EDU_DIR", "chat": "ULTRACHAT_DIR", "math": "FINEMATH_DIR"}
TRAIN_GLOBS = {"web": ["sample/10BT/000_00000.parquet"],
               "chat": ["data/train_sft-00000-of-00003-*.parquet", "data/train_sft-00001-of-00003-*.parquet"],
               "math": ["finemath-4plus/train-00000-of-00064.parquet"]}
HELDOUT_GLOBS = {"web": ["sample/10BT/000_00000.parquet"], "chat": ["data/test_sft-*.parquet"],
                 "math": ["finemath-4plus/train-00000-of-00064.parquet"]}
CHAT_SYSTEM = "You are a helpful assistant."
THINK_EMPTY = "<think>\n\n</think>\n\n"


def sha256_file(path, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- n-gram decontamination
def ngram_hashes(ids: np.ndarray, n: int = NGRAM) -> np.ndarray:
    """uint64 polynomial hash of every length-n token window (wrapping arithmetic; len(ids) - n + 1 values)."""
    ids = np.asarray(ids, dtype=np.uint64)
    m = len(ids) - n + 1
    if m <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(n):
            h = h * _P + ids[j:j + m]
    return h


def dev_texts(dev_rows_gz: str):
    """state + canonical questions JSON of every DEV row (the readout's whole input text, minus template)."""
    with gzip.open(dev_rows_gz, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            yield str(r.get("state") or "")
            yield json.dumps(r.get("questions") or {}, sort_keys=True, ensure_ascii=False)


def build_ngram_set(tok, texts, n: int = NGRAM, batch: int = 256) -> np.ndarray:
    buf, parts = [], []

    def flush():
        if buf:
            for x in tok(buf, add_special_tokens=False)["input_ids"]:
                parts.append(ngram_hashes(np.asarray(x, dtype=np.int64), n))
            buf.clear()

    for t in texts:
        if t:
            buf.append(t)
        if len(buf) >= batch:
            flush()
    flush()
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.uint64)


def contaminated(docs_ids: list, ngram_set: np.ndarray, n: int = NGRAM) -> np.ndarray:
    """Boolean per doc: shares >= 1 n-gram with ``ngram_set`` (sorted unique uint64)."""
    out = np.zeros(len(docs_ids), dtype=bool)
    if len(ngram_set) == 0:
        return out
    for k, ids in enumerate(docs_ids):
        h = ngram_hashes(np.asarray(ids, dtype=np.int64), n)
        if len(h):
            pos = np.searchsorted(ngram_set, h)
            pos[pos == len(ngram_set)] = 0
            out[k] = bool((ngram_set[pos] == h).any())
    return out


# ----------------------------------------------------------------------------- rendering
def render_chat(messages, rng: random.Random) -> str:
    """Qwen3.5 chat format, thinking disabled: an empty think block opens every assistant turn (jrt-chatml-v1)."""
    out = []
    if rng.random() < 0.5:
        out.append(f"<|im_start|>system\n{CHAT_SYSTEM}<|im_end|>\n")
    for m in messages or []:
        role, content = m.get("role"), (m.get("content") or "").strip()
        if role not in ("user", "assistant", "system") or not content:
            continue
        if role == "assistant":
            content = THINK_EMPTY + content
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def doc_texts(src: str, table) -> list[str]:
    if src == "chat":
        rng = random.Random(0)
        return [render_chat(m, rng) for m in table.column("messages").to_pylist()]
    return [t for t in table.column("text").to_pylist() if t]


def _files(root: str, globs) -> list[str]:
    out = []
    for g in globs:
        hits = sorted(glob.glob(os.path.join(root, g)))
        if len(hits) != 1:
            raise FileNotFoundError(f"{root}/{g}: {len(hits)} matches (need exactly 1)")
        out += hits
    return out


# ----------------------------------------------------------------------------- build
def _row_groups(files, heldout_last: bool):
    """(file, row_group) in read order. heldout_last: the last row group of each file is reserved (held-out)."""
    import pyarrow.parquet as pq

    for fp in files:
        n = pq.ParquetFile(fp).num_row_groups
        for rg in range(n - 1 if heldout_last else n):
            yield fp, rg


def _encode_stream(tok, src, units, quota, ngram_set, fo, eos, stats, batch_docs=512):
    """Tokenise docs of ``units`` (file, row_group) into ``fo`` until ``quota`` tokens; returns tokens written."""
    import pyarrow.parquet as pq

    n = 0
    cols = ["messages"] if src == "chat" else ["text"]
    pfs = {}
    for fp, rg in units:
        if fp not in pfs:
            pfs[fp] = pq.ParquetFile(fp)
        texts = doc_texts(src, pfs[fp].read_row_group(rg, columns=cols))
        stats["row_groups"] += 1
        for s in range(0, len(texts), batch_docs):
            ids = tok(texts[s:s + batch_docs], add_special_tokens=False)["input_ids"]
            bad = contaminated(ids, ngram_set)
            stats["docs_seen"] += len(ids)
            stats["docs_dropped_dev_ngram"] += int(bad.sum())
            for x, b in zip(ids, bad):
                if b or not x:
                    continue
                a = np.asarray(x + [eos], dtype=np.uint32)
                fo.write(a.tobytes())
                n += len(a)
                stats["docs_kept"] += 1
                if n >= quota:
                    return n
    return n


def build(cfg: dict, dev_rows: str, out_dir: str, tok=None, roots: dict | None = None) -> dict:
    t0 = time.time()
    mc = cfg["mix"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "mix.json"
    if meta_p.exists():
        return json.loads(meta_p.read_text(encoding="utf-8"))
    if tok is None:
        from transformers import AutoTokenizer

        from convert.model import resolve_model_dir

        tok = AutoTokenizer.from_pretrained(resolve_model_dir(os.path.expandvars(cfg["model_dir"])))
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    roots = roots or {s: os.environ[SOURCE_ENV[s]] for s in SOURCES}
    ng = build_ngram_set(tok, dev_texts(dev_rows))
    meta = {"ngram": NGRAM, "dev_rows_sha256": sha256_file(dev_rows), "dev_ngrams": int(len(ng)), "eos_id": int(eos),
            "train_tokens_target": mc["train_tokens"], "shares": mc["shares"], "sources": {}, "heldout": {}}
    total = mc["train_tokens"]
    tmp = out / "train.u32.tmp"
    with open(tmp, "wb") as fo:
        for src in SOURCES:
            quota = int(round(total * mc["shares"][src]))
            files = _files(roots[src], TRAIN_GLOBS[src])
            st = {"row_groups": 0, "docs_seen": 0, "docs_kept": 0, "docs_dropped_dev_ngram": 0}
            n = _encode_stream(tok, src, _row_groups(files, heldout_last=src != "chat"), quota, ng, fo, eos, st)
            if n < quota:
                raise SystemExit(f"{src}: only {n} tokens before the held-out row group (quota {quota})")
            meta["sources"][src] = {"tokens": n, "quota": quota, **st,
                                    "files": {os.path.relpath(f, roots[src]).replace(os.sep, "/"): sha256_file(f)
                                              for f in files}}
    os.replace(tmp, out / "train.u32")
    for src in SOURCES:
        import pyarrow.parquet as pq

        files = _files(roots[src], HELDOUT_GLOBS[src])
        if src == "chat":
            units = list(_row_groups(files, heldout_last=False))
        else:
            units = [(files[0], pq.ParquetFile(files[0]).num_row_groups - 1)]
        st = {"row_groups": 0, "docs_seen": 0, "docs_kept": 0, "docs_dropped_dev_ngram": 0}
        p = out / f"heldout_{src}.u32"
        with open(str(p) + ".tmp", "wb") as fo:
            n = _encode_stream(tok, src, units, mc["heldout_tokens"][src], ng, fo, eos, st)
        os.replace(str(p) + ".tmp", p)
        meta["heldout"][src] = {"tokens": n, **st, "units": [[os.path.relpath(f, roots[src]).replace(os.sep, "/"), rg]
                                                              for f, rg in units]}
    meta["train_tokens"] = int(os.path.getsize(out / "train.u32") // 4)
    meta["train_sha256"] = sha256_file(out / "train.u32")
    meta["heldout_sha256"] = {s: sha256_file(out / f"heldout_{s}.u32") for s in SOURCES}
    meta["seconds"] = round(time.time() - t0, 1)
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- consumption
class WindowStream:
    """Non-overlapping length-L windows of the train stream in one seeded permutation. The token budget is sized so
    a run never re-reads text; if it ever would, a fresh permutation starts (counted in ``passes``, reported) instead
    of crashing a paid run."""

    def __init__(self, toks, seq_len, batch, seed, device="cuda"):
        self.t, self.L, self.B, self.device, self.seed = toks, seq_len, batch, device, seed
        self.n = (len(toks) - 1) // seq_len
        self.passes = 0
        self._perm()
        self.used = 0

    def _perm(self):
        self.order = np.random.default_rng(self.seed + self.passes).permutation(self.n)
        self.pos = 0

    @property
    def windows_used(self):
        return self.used

    def skip(self, n_windows):
        for _ in range(max(0, n_windows) // self.B):
            self._take()
        for _ in range(max(0, n_windows) % self.B):
            self._take(1)

    def _take(self, b=None):
        b = b or self.B
        if self.pos + b > len(self.order):
            self.passes += 1
            self._perm()
        idx = self.order[self.pos:self.pos + b]
        self.pos += b
        self.used += b
        return idx

    def __next__(self):
        import torch

        idx = self._take()
        x = np.stack([np.asarray(self.t[i * self.L:(i + 1) * self.L], dtype=np.int64) for i in idx])
        return torch.from_numpy(x).to(self.device, non_blocking=True)

    def __iter__(self):
        return self


def main(argv=None):
    ap = argparse.ArgumentParser(prog="convert.mixdata")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--config", required=True)
    b.add_argument("--dev-rows", required=True)
    b.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    meta = build(json.loads(Path(a.config).read_text(encoding="utf-8")), a.dev_rows, a.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"} | {
        "sources": {s: {k: v for k, v in d.items() if k != "files"} for s, d in meta["sources"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
