"""Generic distillation text for Q9-AF v2 (jobs/q9af_v2): a pinned public mixture, DEV-decontaminated, never re-used.

    python -m convert.mixdata build --config configs/q9af_v2_h100.json --dev-rows <dev-rows.jsonl.gz> --out <dir> [--decontam F]

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

Decontamination (pre-registered, reported), selected by ``mix.decontam`` in the config (``DECONTAM_*`` below):
  dev13-v1      (job-w3c2e, recorded in configs/q9af_v2_h100.json): every document sharing ANY token 13-gram with any
                DEV row (state + canonical questions JSON, ``slices/dev-rows.jsonl.gz``, all 12,104 rows: every Index
                benchmark incl. the frozen families' DEV rows) is dropped. The DEV rows are used only to build this
                n-gram filter, never as text. DEV-ONLY: TEST, held-out and attack templates are not filtered.
  full-eval-v1  (the DEFAULT for any config without ``mix.decontam``, i.e. every new conversion): dev13-v1, plus every
                document sharing ANY distinctive word 13-gram with the pinned Index eval-hash index (DEV + TEST +
                held-out units, frozen families included; ``train.doc_overlap_guard``: hashes only, sha-pinned in
                ``data/manifests/doc_overlap/index-pin.json``) is dropped, plus every document containing an eval
                attack template fragment (``train.attack_guard``, base64 payloads included). "Distinctive" = the gram
                is in at least one non-allowlisted eval unit and in fewer than ``TEMPLATE_EVAL_DF`` (100) eval rows
                (eval-side frames such as the HellaSwag prompt are not evidence of a copied document). Drops are
                counted per reason (``docs_dropped_dev_ngram`` / ``docs_dropped_eval_index`` / ``docs_dropped_attack``).
A build refuses an unknown filter name; mix.json records the filter, its rule text and the index / template shas.

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

#: decontamination filters (``mix.decontam``). dev13-v1 is what job-w3c2e ran; a config that names no filter gets the
#: full one, so a new conversion can only filter DEV-only by saying so.
DECONTAM_DEV = "dev13-v1"
DECONTAM_FULL = "full-eval-v1"
DECONTAM_DEFAULT = DECONTAM_FULL
DECONTAM_RULES = {
    DECONTAM_DEV: "drop a doc sharing any Qwen-token 13-gram with a DEV row (state + questions JSON)",
    DECONTAM_FULL: ("dev13-v1 + drop a doc sharing any word 13-gram (train.doc_overlap_guard normalise/hash, the doc as "
                    "one leaf) with the pinned Index eval-hash index (DEV + TEST + held-out) that is in a non-allowlisted "
                    "eval unit and in < 100 eval rows + drop a doc containing any eval attack template fragment "
                    "(train.attack_guard, base64 decodes included)"),
}
#: reason codes of ``eval_flags`` (bit mask)
FLAG_EVAL, FLAG_ATTACK = 1, 2


def decontam_of(mc: dict) -> str:
    name = mc.get("decontam", DECONTAM_DEFAULT)
    if name not in DECONTAM_RULES:
        raise SystemExit(f"mix.decontam {name!r}: unknown filter (one of {sorted(DECONTAM_RULES)})")
    return name


def eval_flags(texts, idx=None, matcher=None) -> np.ndarray:
    """full-eval-v1 per doc: FLAG_EVAL if the doc shares a distinctive word 13-gram with the eval-hash index,
    FLAG_ATTACK if it contains an eval attack template fragment (uint8 bit mask, one per text)."""
    from train import attack_guard as A
    from train import doc_overlap_guard as G

    idx = idx if idx is not None else G.get_index()
    matcher = matcher or A.eval_matcher()
    out = np.zeros(len(texts), dtype=np.uint8)
    for k, t in enumerate(texts):
        if not t:
            continue
        pos = idx.lookup(G.text_grams([t]))
        if len(pos) and bool((((idx.fl[pos] & 1) > 0) & (idx.df[pos] < G.TEMPLATE_EVAL_DF)).any()):
            out[k] |= FLAG_EVAL
        if A.record_hits({"state": t}, matcher):
            out[k] |= FLAG_ATTACK
    return out


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


def doc_rows(src: str, table) -> tuple[list[int], list[str]]:
    """(row index in the row group, text) of every document of a row group, in read order (empty web/math rows are
    skipped, as ever; a chat row always renders, consuming the row group's seeded system-prompt draw)."""
    if src == "chat":
        rng = random.Random(0)
        texts = [render_chat(m, rng) for m in table.column("messages").to_pylist()]
        return list(range(len(texts))), texts
    rows = [(i, t) for i, t in enumerate(table.column("text").to_pylist()) if t]
    return [i for i, _ in rows], [t for _, t in rows]


def doc_texts(src: str, table) -> list[str]:
    return doc_rows(src, table)[1]


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


def _new_stats(decontam: str) -> dict:
    st = {"row_groups": 0, "docs_seen": 0, "docs_kept": 0, "docs_dropped_dev_ngram": 0}
    if decontam == DECONTAM_FULL:
        st |= {"docs_dropped_eval_index": 0, "docs_dropped_attack": 0}
    return st


def _encode_stream(tok, src, units, quota, ngram_set, fo, eos, stats, batch_docs=512, doc_filter=None, on_doc=None,
                   stream="train"):
    """Tokenise docs of ``units`` (file, row_group) into ``fo`` until ``quota`` tokens; returns tokens written.

    doc_filter(texts) -> uint8 FLAG_* mask per text (full-eval-v1; None = dev13-v1 only). on_doc(stream, src, fp, rg,
    row, text, n_tokens, dropped) is called for every doc in read order up to the quota (dropped: None when written,
    else "dev_ngram" / "eval_index" / "attack" / "empty"); it never changes what is written."""
    import pyarrow.parquet as pq

    n = 0
    cols = ["messages"] if src == "chat" else ["text"]
    pfs = {}
    for fp, rg in units:
        if fp not in pfs:
            pfs[fp] = pq.ParquetFile(fp)
        rows, texts = doc_rows(src, pfs[fp].read_row_group(rg, columns=cols))
        stats["row_groups"] += 1
        for s in range(0, len(texts), batch_docs):
            chunk = texts[s:s + batch_docs]
            ids = tok(chunk, add_special_tokens=False)["input_ids"]
            bad = contaminated(ids, ngram_set)
            flags = doc_filter(chunk) if doc_filter is not None else None
            stats["docs_seen"] += len(ids)
            stats["docs_dropped_dev_ngram"] += int(bad.sum())
            for k, (x, b) in enumerate(zip(ids, bad)):
                why = "dev_ngram" if b else None
                if why is None and flags is not None and flags[k]:
                    why = "eval_index" if flags[k] & FLAG_EVAL else "attack"
                    stats[f"docs_dropped_{why}"] += 1
                if why is None and not x:
                    why = "empty"
                if why is not None:
                    if on_doc is not None:
                        on_doc(stream, src, fp, rg, rows[s + k], chunk[k], 0, why)
                    continue
                a = np.asarray(x + [eos], dtype=np.uint32)
                fo.write(a.tobytes())
                n += len(a)
                stats["docs_kept"] += 1
                if on_doc is not None:
                    on_doc(stream, src, fp, rg, rows[s + k], chunk[k], len(a), None)
                if n >= quota:
                    return n
    return n


def full_filter(idx=None):
    """The full-eval-v1 doc filter + its identity for mix.json (the pinned eval index and the attack template set)."""
    from train import attack_guard as A
    from train import doc_overlap_guard as G

    idx = idx if idx is not None else G.get_index()
    m = A.eval_matcher()
    ident = {"eval_index_sha256": idx.meta.get("index_sha256"), "eval_index_rule": G.RULE,
             "eval_index_rows_by_split": idx.meta.get("rows_by_split"), "attack_rule": A.RULE,
             "attack_templates_sha256": A.eval_templates_sha256(), "template_eval_df": G.TEMPLATE_EVAL_DF}
    return (lambda texts: eval_flags(texts, idx, m)), ident


def build(cfg: dict, dev_rows: str, out_dir: str, tok=None, roots: dict | None = None, on_doc=None,
          decontam: str | None = None, eval_index=None) -> dict:
    """``decontam`` overrides ``mix.decontam`` (the config's recorded filter; absent = DECONTAM_DEFAULT)."""
    t0 = time.time()
    mc = cfg["mix"]
    dc = decontam_of({"decontam": decontam} if decontam else mc)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_p = out / "mix.json"
    if meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("decontam", DECONTAM_DEV) != dc:
            raise SystemExit(f"{meta_p} was built with decontam {meta.get('decontam', DECONTAM_DEV)!r}, not {dc!r}")
        return meta
    doc_filter, filt_ident = full_filter(eval_index) if dc == DECONTAM_FULL else (None, {})
    if tok is None:
        from transformers import AutoTokenizer

        from convert.model import resolve_model_dir

        tok = AutoTokenizer.from_pretrained(resolve_model_dir(os.path.expandvars(cfg["model_dir"])))
    eos = tok.convert_tokens_to_ids("<|endoftext|>")
    roots = roots or {s: os.environ[SOURCE_ENV[s]] for s in SOURCES}
    ng = build_ngram_set(tok, dev_texts(dev_rows))
    meta = {"ngram": NGRAM, "dev_rows_sha256": sha256_file(dev_rows), "dev_ngrams": int(len(ng)), "eos_id": int(eos),
            "train_tokens_target": mc["train_tokens"], "shares": mc["shares"], "sources": {}, "heldout": {}}
    if dc != DECONTAM_DEV:  # a dev13-v1 mix.json stays byte-for-byte the shape job-w3c2e wrote
        meta |= {"decontam": dc, "decontam_rule": DECONTAM_RULES[dc], **filt_ident}
    total = mc["train_tokens"]
    tmp = out / "train.u32.tmp"
    with open(tmp, "wb") as fo:
        for src in SOURCES:
            quota = int(round(total * mc["shares"][src]))
            files = _files(roots[src], TRAIN_GLOBS[src])
            st = _new_stats(dc)
            n = _encode_stream(tok, src, _row_groups(files, heldout_last=src != "chat"), quota, ng, fo, eos, st,
                               doc_filter=doc_filter, on_doc=on_doc, stream="train")
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
        st = _new_stats(dc)
        p = out / f"heldout_{src}.u32"
        with open(str(p) + ".tmp", "wb") as fo:
            n = _encode_stream(tok, src, units, mc["heldout_tokens"][src], ng, fo, eos, st, doc_filter=doc_filter,
                               on_doc=on_doc, stream=f"heldout_{src}")
        os.replace(str(p) + ".tmp", p)
        meta["heldout"][src] = {"tokens": n, **st, "units": [[os.path.relpath(f, roots[src]).replace(os.sep, "/"), rg]
                                                              for f, rg in units]}
    meta["train_tokens"] = int(os.path.getsize(out / "train.u32") // 4)
    meta["train_sha256"] = sha256_file(out / "train.u32")
    meta["heldout_sha256"] = {s: sha256_file(out / f"heldout_{s}.u32") for s in SOURCES}
    meta["seconds"] = round(time.time() - t0, 1)
    meta_p.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# ----------------------------------------------------------------------------- per-window manifest
#: convert/joint.py: WindowStream(train, L, B, seed=cfg["seed"] + 7)
JOINT_SEED_OFFSET = 7
WINDOWS_FORMAT = "q9af-window-manifest-v1"


class WindowManifest:
    """``build``'s on_doc hook: the identity and token span of every WRITTEN doc, in stream order, so a run's
    training windows can be audited later (window id -> source docs + token span) without the text.

    Doc ids are ``<stream>/<src>/f<file idx>/rg<row group>/r<row>`` with file indices in first-written order: the
    same ids ``jobs.q9af_corpus_check.corpus_check`` reports. Train window w = tokens [w*L, (w+1)*L) of train.u32
    (``WindowStream``: n_windows = (train_tokens - 1) // L); ``rank[w]`` = its position in the first permutation
    (seed + 7), so a window was trained iff rank < the run's windows_used (results.json)."""

    def __init__(self, roots: dict):
        self.roots = roots
        self.files: dict[tuple, int] = {}
        self.docs: dict[str, list[tuple]] = {}
        self.offsets: dict[str, int] = {}
        self.dropped: dict[str, int] = {}

    def __call__(self, stream, src, fp, rg, row, text, ntok, dropped):
        if dropped is not None:
            k = f"{stream}/{src}/{dropped}"
            self.dropped[k] = self.dropped.get(k, 0) + 1
            return
        fi = self.files.setdefault((src, fp), len(self.files))
        off = self.offsets.get(stream, 0)
        self.offsets[stream] = off + int(ntok)
        self.docs.setdefault(stream, []).append((SOURCES.index(src), fi, int(rg), int(row), off, int(ntok)))

    def _cols(self, stream: str) -> dict:
        d = np.asarray(self.docs.get(stream, []), dtype=np.int64).reshape(-1, 6)
        return {"src": d[:, 0].tolist(), "file": d[:, 1].tolist(), "rg": d[:, 2].tolist(), "row": d[:, 3].tolist(),
                "off": d[:, 4].tolist(), "ntok": d[:, 5].tolist()}

    def manifest(self, meta: dict, seq_len: int, seed: int) -> dict:
        tr = np.asarray(self.docs.get("train", []), dtype=np.int64).reshape(-1, 6)
        off, ntok = tr[:, 4], tr[:, 5]
        total = int(off[-1] + ntok[-1]) if len(tr) else 0
        if total != int(meta["train_tokens"]):
            raise SystemExit(f"window manifest: docs cover {total} tokens, train.u32 has {meta['train_tokens']}")
        n_win = max(0, (total - 1) // seq_len)
        starts = np.arange(n_win, dtype=np.int64) * seq_len
        end = off + ntok
        lo = np.searchsorted(end, starts, side="right")           # first doc ending after the window start
        hi = np.searchsorted(off, starts + seq_len, side="left") - 1  # last doc starting before the window end
        perm_seed = int(seed) + JOINT_SEED_OFFSET
        perm = np.random.default_rng(perm_seed).permutation(n_win)
        rank = np.empty(n_win, np.int64)
        rank[perm] = np.arange(n_win)
        files = [None] * len(self.files)
        for (src, fp), i in self.files.items():
            files[i] = f"{src}/" + os.path.relpath(fp, self.roots[src]).replace(os.sep, "/")
        return {"format": WINDOWS_FORMAT, "doc_id": "<stream>/<SOURCES[src]>/f<file>/rg<rg>/r<row>",
                "sources": list(SOURCES), "files": files, "decontam": meta.get("decontam", DECONTAM_DEV),
                "train_sha256": meta["train_sha256"], "train_tokens": total, "dropped": self.dropped,
                "train": {"docs": self._cols("train"),
                          "windows": {"seq_len": int(seq_len), "n_windows": int(n_win), "perm_seed": perm_seed,
                                      "span": "window w = train tokens [w*seq_len, (w+1)*seq_len)",
                                      "doc_lo": lo.tolist(), "doc_hi": hi.tolist(), "rank": rank.tolist()}},
                "heldout": {s.split("_", 1)[1]: {"docs": self._cols(s)} for s in sorted(self.docs) if s != "train"}}

    def write(self, path, meta: dict, seq_len: int, seed: int) -> dict:
        man = self.manifest(meta, seq_len, seed)
        tmp = str(path) + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=9) as f:
            json.dump(man, f, separators=(",", ":"))
        os.replace(tmp, path)
        return {"windows_manifest": os.path.basename(str(path)), "windows_manifest_sha256": sha256_file(path),
                "n_windows": man["train"]["windows"]["n_windows"], "train_docs": len(man["train"]["docs"]["off"])}


def load_eval_index_dir(d: str):
    """The pinned eval-hash index from a shipped dir (hashes only): rule, allowlist and array sha256 must match the
    committed pin (data/manifests/doc_overlap/index-pin.json). Never rebuilds, never reads the suite."""
    from train import doc_overlap_guard as G

    pin = G.read_pin()
    if not pin:
        raise SystemExit("no data/manifests/doc_overlap/index-pin.json")
    if pin.get("rule") != G.RULE or pin.get("allowlist_sha256") != G.allowlist_sha256():
        raise SystemExit("index pin was made under a different rule / allowlist")
    return G.load_index_dir(Path(d), pin["index_sha256"])


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
    b.add_argument("--decontam", choices=sorted(DECONTAM_RULES),
                   help=f"override the config's mix.decontam (absent there: {DECONTAM_DEFAULT})")
    b.add_argument("--eval-index-dir", help="full-eval-v1: load the shipped, pin-checked eval-hash index from here")
    b.add_argument("--windows-out", help="also write the per-window manifest (gz JSON, WindowManifest) here")
    a = ap.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    idx = load_eval_index_dir(a.eval_index_dir) if a.eval_index_dir else None
    wm = None
    if a.windows_out:
        if (Path(a.out) / "mix.json").exists() and not Path(a.windows_out).exists():
            raise SystemExit(f"{a.out}/mix.json exists without {a.windows_out}: the manifest needs a fresh build")
        wm = WindowManifest({s: os.environ[SOURCE_ENV[s]] for s in SOURCES})
    kw = {k: v for k, v in (("eval_index", idx), ("on_doc", wm)) if v is not None}
    meta = build(cfg, a.dev_rows, a.out, decontam=a.decontam, **kw)
    if wm is not None and wm.docs:
        info = wm.write(a.windows_out, meta, cfg["seq_len"], cfg["seed"])
        meta |= info
        (Path(a.out) / "mix.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "sources"} | {
        "sources": {s: {k: v for k, v in d.items() if k != "files"} for s, d in meta["sources"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
