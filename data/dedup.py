"""Contamination guard: every training record is checked against the FULL Decision Index suite.

Scope of the fingerprint: every row of both suite files (``selected-rows`` and ``added-rows``), all catalogs
(index and display-only), DEV and TEST alike, excluded questions included. Nothing is filtered by slice: the
guard must not need, and does not use, the DEV/TEST split. The fingerprint is built from a snapshot of the
OFFICIAL row files (HLE included; hashes checked against ``EXPECTED_SUITE_SHA256``) copied to
``m2/suite-snapshot``, so a concurrent re-stage of ``suite-0.2`` cannot change it mid-build.

Three matchers, each logged separately (a record is dropped if ANY fires on its item-tier texts; see DedupLog):

  exact     normalised segment hash equality. Segments are every string in state, every instruction and every
            option description, plus each line and each line without a short "Label:" prefix, plus the whole
            state; normalised = NFKC + casefold + non-alphanumerics collapsed; >= 12 chars.
  ngram13   any shared word 13-gram (GPT-3 style contamination test) with an Index row.
  minhash   estimated Jaccard >= MINHASH_THRESHOLD on character 5-gram shingles (64 permutations, LSH with
            16 bands x 4 rows, verified on the full signature) for segments of 12..1000 chars. Catches short
            items with small edits (BANKING77/CLINC utterances, WinoGrande twins), which 13-grams cannot.

Boilerplate: an Index segment or 13-gram that appears in more than BOILERPLATE_ROWS distinct rows is template
text (instruction prefixes, label descriptions shared by every BANKING77 row, ACOS category questions), not an
item, and is left out of the fingerprint. The threshold and the counts it removes are written to the log.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from . import text as T

FP_VERSION = "ssdm-dedup-v2"  # v2: Unicode-aware normalisation (v1 dropped non-Latin text)
BOILERPLATE_ROWS = 100
NGRAM_N = 13
MINHASH_PERMS = 64
MINHASH_BANDS = 16
MINHASH_ROWS = MINHASH_PERMS // MINHASH_BANDS
MINHASH_MAX_CHARS = 1000
MINHASH_THRESHOLD = 0.75
SHINGLE_K = 5
SUITE_FILES = ("selected-rows.jsonl.gz", "added-rows.jsonl.gz")

_rng = np.random.default_rng(20261006)
_A = (_rng.integers(1, 2**63, size=MINHASH_PERMS, dtype=np.uint64) | np.uint64(1))
_B = _rng.integers(0, 2**63, size=MINHASH_PERMS, dtype=np.uint64)
_SHIFT = np.uint64(32)
_MASK32 = np.uint64(0xFFFFFFFF)
_BYTE_P = np.uint64(1099511628211)


# ----------------------------------------------------------------------------------------------- minhash
def _shingle_hashes_batch(texts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Rolling hashes of every byte 5-gram of every text; returns (hashes, segment index per hash)."""
    bufs = [t.encode("utf-8") for t in texts]
    lens = np.fromiter((len(b) for b in bufs), dtype=np.int64, count=len(bufs))
    data = np.frombuffer(b"".join(bufs), dtype=np.uint8).astype(np.uint64)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    k = SHINGLE_K
    total = data.shape[0]
    m = max(total - k + 1, 0)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(k):
            h = h * _BYTE_P + data[i:i + m]
    seg_of_pos = np.repeat(np.arange(len(bufs)), lens)
    pos = np.arange(m)
    valid = (pos + k - 1 < total)
    valid &= seg_of_pos[:m] == seg_of_pos[np.minimum(pos + k - 1, total - 1)]
    hs, seg = h[valid], seg_of_pos[:m][valid]
    # texts shorter than k bytes: hash the whole text once
    short = np.nonzero(lens < k)[0]
    if short.size:
        extra = np.array([T.h64(texts[i]) for i in short], dtype=np.uint64)
        hs = np.concatenate([hs, extra])
        seg = np.concatenate([seg, short])
        order = np.argsort(seg, kind="stable")
        hs, seg = hs[order], seg[order]
    return hs, seg


def minhash_signatures(texts: list[str], batch: int = 4000) -> np.ndarray:
    """(len(texts), MINHASH_PERMS) uint32 signatures. Multiply-shift universal hashing, wrapping uint64."""
    out = np.empty((len(texts), MINHASH_PERMS), dtype=np.uint32)
    for b0 in range(0, len(texts), batch):
        chunk = texts[b0:b0 + batch]
        hs, seg = _shingle_hashes_batch(chunk)
        bounds = np.searchsorted(seg, np.arange(len(chunk)))
        with np.errstate(over="ignore"):
            for p in range(MINHASH_PERMS):
                v = ((hs * _A[p] + _B[p]) >> _SHIFT) & _MASK32
                out[b0:b0 + len(chunk), p] = np.minimum.reduceat(v, bounds).astype(np.uint32)
    return out


def band_keys(sig: np.ndarray) -> np.ndarray:
    """(n, MINHASH_BANDS) uint64 keys; band b hashes its MINHASH_ROWS signature values (band id mixed in)."""
    n = sig.shape[0]
    keys = np.empty((n, MINHASH_BANDS), dtype=np.uint64)
    s = sig.astype(np.uint64)
    with np.errstate(over="ignore"):
        for b in range(MINHASH_BANDS):
            k = np.full(n, np.uint64(b + 1) * np.uint64(0x9E3779B97F4A7C15), dtype=np.uint64)
            for r in range(MINHASH_ROWS):
                k = k * _BYTE_P + s[:, b * MINHASH_ROWS + r]
            keys[:, b] = k
    return keys


# ----------------------------------------------------------------------------------------------- suite iteration
def iter_suite_rows(suite_dir: Path) -> Iterator[dict]:
    for fn in SUITE_FILES:
        with gzip.open(Path(suite_dir) / fn, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _sha256_gunzip(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


# Official uncompressed hashes of the 0.2.1 row files (kit editions.py; harness/config.py). The fingerprint is
# built only from files matching these (the official selected-rows includes HLE's 513 rows).
EXPECTED_SUITE_SHA256 = {
    "selected-rows.jsonl.gz": "b2b56d6fb636837ca469e689087bdbf373dda8de7638aa2da6793e6eda0792d5",
    "added-rows.jsonl.gz": "7429f3c9cdddb772c1cfc42bb2a45e8516b0032152b746e6929f1c8b52f4ce89",
}


# ----------------------------------------------------------------------------------------------- fingerprint
@dataclass
class Fingerprint:
    seg_hash: np.ndarray            # sorted uint64, non-boilerplate segments
    seg_cat: np.ndarray             # int32 catalog id of first row with that segment
    ng_hash: np.ndarray             # sorted uint64, non-boilerplate 13-grams
    ng_cat: np.ndarray              # int32
    mh_sig: np.ndarray              # (m, perms) uint32 signatures of short non-boilerplate segments
    mh_cat: np.ndarray              # int32
    mh_band_sorted: list = field(default_factory=list)   # per band: (sorted keys, row index)
    meta: dict = field(default_factory=dict)

    # -- build --------------------------------------------------------------------------------------
    @classmethod
    def build(cls, rows: Iterable[dict], source_meta: dict | None = None, log=print) -> "Fingerprint":
        t0 = time.time()
        seg_rows: dict[int, int] = collections.Counter()
        seg_first_cat: dict[int, int] = {}
        seg_text: dict[int, str] = {}
        ng_chunks, ng_cat_chunks = [], []
        n = 0
        cats = collections.Counter()
        for r in rows:
            n += 1
            ev = r.get("_evaluation", {})
            cat = int(ev.get("catalog_id", -1))
            cats[cat] += 1
            for s in T.segments(r.get("state"), r.get("questions")):
                hv = T.h64(s)
                seg_rows[hv] += 1
                if hv not in seg_first_cat:
                    seg_first_cat[hv] = cat
                    if len(s) <= MINHASH_MAX_CHARS:
                        seg_text[hv] = s
            hs = []
            for t in T.payload_texts(r.get("state"), r.get("questions")):
                ws = t.split()
                if len(ws) >= NGRAM_N:
                    hs.append(T.ngram_hashes(T.word_ids(ws), NGRAM_N))
            if hs:
                u = np.unique(np.concatenate(hs))
                ng_chunks.append(u)
                ng_cat_chunks.append(np.full(u.size, cat, dtype=np.int32))
            if n % 25000 == 0:
                log(f"  fingerprint: {n} rows, {len(seg_rows)} segments, {time.time() - t0:.0f}s")
        # segments
        hv_all = np.fromiter(seg_rows.keys(), dtype=np.uint64, count=len(seg_rows))
        cnt_all = np.fromiter(seg_rows.values(), dtype=np.int64, count=len(seg_rows))
        keep = cnt_all <= BOILERPLATE_ROWS
        seg_hash = hv_all[keep]
        order = np.argsort(seg_hash)
        seg_hash = seg_hash[order]
        seg_cat = np.array([seg_first_cat[int(x)] for x in seg_hash], dtype=np.int32)
        # 13-grams
        if ng_chunks:
            allng = np.concatenate(ng_chunks)
            allcat = np.concatenate(ng_cat_chunks)
            del ng_chunks, ng_cat_chunks
            order = np.argsort(allng, kind="stable")
            allng, allcat = allng[order], allcat[order]
            uniq, first, counts = np.unique(allng, return_index=True, return_counts=True)
            keep_ng = counts <= BOILERPLATE_ROWS
            ng_hash, ng_cat = uniq[keep_ng], allcat[first][keep_ng]
            ng_total, ng_boiler = int(uniq.size), int((~keep_ng).sum())
            del allng, allcat
        else:
            ng_hash = np.empty(0, np.uint64); ng_cat = np.empty(0, np.int32); ng_total = ng_boiler = 0
        # minhash over short non-boilerplate segments
        mh_hashes = [int(x) for x in seg_hash if int(x) in seg_text]
        mh_texts = [seg_text[x] for x in mh_hashes]
        mh_cat = np.array([seg_first_cat[x] for x in mh_hashes], dtype=np.int32)
        log(f"  fingerprint: minhash over {len(mh_texts)} segments")
        mh_sig = minhash_signatures(mh_texts)
        fp = cls(seg_hash=seg_hash, seg_cat=seg_cat, ng_hash=ng_hash, ng_cat=ng_cat, mh_sig=mh_sig, mh_cat=mh_cat)
        fp.meta = {
            "version": FP_VERSION,
            "rows": n,
            "rows_by_catalog": {str(k): v for k, v in sorted(cats.items())},
            "segments_unique": int(hv_all.size),
            "segments_boilerplate_removed": int((~keep).sum()),
            "segments_indexed": int(seg_hash.size),
            "ngram13_unique": ng_total,
            "ngram13_boilerplate_removed": ng_boiler,
            "ngram13_indexed": int(ng_hash.size),
            "minhash_segments": int(mh_sig.shape[0]),
            "params": {"boilerplate_rows": BOILERPLATE_ROWS, "ngram_n": NGRAM_N, "minhash_perms": MINHASH_PERMS,
                       "minhash_bands": MINHASH_BANDS, "minhash_max_chars": MINHASH_MAX_CHARS,
                       "minhash_threshold": MINHASH_THRESHOLD, "shingle": f"byte {SHINGLE_K}-gram",
                       "min_segment_chars": T.MIN_SEGMENT_CHARS},
            "build_seconds": round(time.time() - t0, 1),
            **(source_meta or {}),
        }
        fp._index_bands()
        return fp

    @classmethod
    def build_from_suite(cls, suite_dir: Path, log=print, allow_unofficial: bool = False) -> "Fingerprint":
        suite_dir = Path(suite_dir)
        before = {fn: _sha256_gunzip(suite_dir / fn) for fn in SUITE_FILES}
        fp = cls.build(iter_suite_rows(suite_dir), {"suite_dir": str(suite_dir),
                                                    "suite_files_uncompressed_sha256": before}, log=log)
        after = {fn: _sha256_gunzip(suite_dir / fn) for fn in SUITE_FILES}
        if before != after:  # another process rewrote the suite mid-build
            raise RuntimeError(f"suite files changed during the build: {before} -> {after}")
        if before != EXPECTED_SUITE_SHA256 and not allow_unofficial:
            raise RuntimeError(f"suite files are not the official 0.2.1 rows: {before}")
        return fp

    def _index_bands(self):
        keys = band_keys(self.mh_sig) if self.mh_sig.shape[0] else np.empty((0, MINHASH_BANDS), np.uint64)
        self.mh_band_sorted = []
        for b in range(MINHASH_BANDS):
            order = np.argsort(keys[:, b])
            self.mh_band_sorted.append((keys[order, b], order))

    # -- persistence ------------------------------------------------------------------------------------
    def save(self, path: Path) -> str:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, seg_hash=self.seg_hash, seg_cat=self.seg_cat, ng_hash=self.ng_hash, ng_cat=self.ng_cat,
                 mh_sig=self.mh_sig, mh_cat=self.mh_cat)
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=1), encoding="utf-8")
        return _sha256_file(path)

    @classmethod
    def load(cls, path: Path) -> "Fingerprint":
        path = Path(path)
        z = np.load(path, allow_pickle=False)
        fp = cls(seg_hash=z["seg_hash"], seg_cat=z["seg_cat"], ng_hash=z["ng_hash"], ng_cat=z["ng_cat"],
                 mh_sig=z["mh_sig"], mh_cat=z["mh_cat"])
        fp.meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        fp.meta["npz_sha256"] = _sha256_file(path)
        fp._index_bands()
        return fp

    # -- queries ----------------------------------------------------------------------------------------
    @staticmethod
    def _lookup(sorted_arr: np.ndarray, q: np.ndarray) -> np.ndarray:
        if sorted_arr.size == 0 or q.size == 0:
            return np.full(q.shape, -1, dtype=np.int64)
        i = np.searchsorted(sorted_arr, q)
        i = np.minimum(i, sorted_arr.size - 1)
        return np.where(sorted_arr[i] == q, i, -1)

    def check(self, state: Any, questions: dict) -> dict:
        """Matches of one record against the fingerprint: {matcher: sorted list of Index catalog ids}.

        Results are cached per normalised segment / text (retrieval records reuse the same passages)."""
        if not hasattr(self, "_seg_cache"):
            self._seg_cache, self._txt_cache = {}, {}
        hits: dict[str, set] = {"exact": set(), "ngram13": set(), "minhash": set()}
        segs = [s for s in T.segments(state, questions)]
        todo = [s for s in segs if s not in self._seg_cache]
        if todo:
            q = np.array([T.h64(s) for s in todo], dtype=np.uint64)
            idx = self._lookup(self.seg_hash, q)
            short_i = [j for j, s in enumerate(todo) if len(s) <= MINHASH_MAX_CHARS]
            mh: dict[int, set] = {}
            if short_i and self.mh_sig.shape[0]:
                sig = minhash_signatures([todo[j] for j in short_i])
                for jj, cat in self._minhash_hits(sig):
                    mh.setdefault(short_i[jj], set()).add(cat)
            for j, s in enumerate(todo):
                ex = (int(self.seg_cat[idx[j]]),) if idx[j] >= 0 else ()
                self._seg_cache[s] = (ex, tuple(sorted(mh.get(j, ()))))
        for s in segs:
            ex, mh_c = self._seg_cache[s]
            hits["exact"].update(ex)
            hits["minhash"].update(mh_c)
        for t in T.payload_texts(state, questions):
            if t not in self._txt_cache:
                ws = t.split()
                cats: tuple = ()
                if len(ws) >= NGRAM_N:
                    q = np.unique(T.ngram_hashes(T.word_ids(ws), NGRAM_N))
                    idx = self._lookup(self.ng_hash, q)
                    cats = tuple(sorted({int(self.ng_cat[i]) for i in idx[idx >= 0]}))
                self._txt_cache[t] = cats
            hits["ngram13"].update(self._txt_cache[t])
        if len(self._seg_cache) > 2_000_000:
            self._seg_cache.clear()
        if len(self._txt_cache) > 500_000:
            self._txt_cache.clear()
        return {k: sorted(v) for k, v in hits.items()}

    def _minhash_hits(self, sig: np.ndarray):
        """Yield (query row, catalog) for every Index segment with estimated Jaccard >= threshold."""
        keys = band_keys(sig)
        cand: dict[int, set] = {}
        for b, (skeys, order) in enumerate(self.mh_band_sorted):
            lo = np.searchsorted(skeys, keys[:, b], side="left")
            hi = np.searchsorted(skeys, keys[:, b], side="right")
            for j in np.nonzero(hi > lo)[0]:
                cand.setdefault(int(j), set()).update(order[lo[j]:hi[j]].tolist())
        for j, cs in cand.items():
            c = np.fromiter(cs, dtype=np.int64)
            est = (self.mh_sig[c] == sig[j]).mean(axis=1)
            for i in c[est >= MINHASH_THRESHOLD]:
                yield j, int(self.mh_cat[i])


# ----------------------------------------------------------------------------------------------- guard
class DedupLog:
    """Per-source counts; the JSON form is the receipt.

    Two tiers per record. ITEM tier = the texts that carry the item (utterance, premise/hypothesis, question,
    MCQ options, gold passage; the loader declares them): any matcher firing drops the record. FULL tier = the
    whole engine payload (adds shared label sets, tool catalogs, retrieval negatives, padding, instructions,
    injections): matches that are not also item-tier matches are context overlap that carries no Index label
    (INDEX_NOTES section 7) and are logged, not dropped."""

    def __init__(self, fp_meta: dict):
        self.fp_meta = fp_meta
        self.by_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.pairs: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.examples: dict[str, list] = collections.defaultdict(list)
        self.ctx_examples: dict[str, list] = collections.defaultdict(list)

    def add(self, source: str, rec_id: str, item_hits: dict, full_hits: dict | None = None) -> bool:
        c = self.by_source[source]
        c["checked"] += 1
        fired = [k for k, v in item_hits.items() if v]
        for k in fired:
            c[f"item_{k}"] += 1
            for cat in item_hits[k]:
                self.pairs[source][f"item_{k}:{cat}"] += 1
        if fired:
            c["dropped"] += 1
            if len(self.examples[source]) < 5:
                self.examples[source].append({"id": rec_id, "item_hits": item_hits})
            return False
        c["kept"] += 1
        if full_hits and any(full_hits.values()):
            c["kept_with_context_overlap"] += 1
            for k, v in full_hits.items():
                if v:
                    c[f"context_{k}"] += 1
                    for cat in v:
                        self.pairs[source][f"context_{k}:{cat}"] += 1
            if len(self.ctx_examples[source]) < 3:
                self.ctx_examples[source].append({"id": rec_id, "context_hits": full_hits})
        return True

    def to_json(self) -> dict:
        tot = collections.Counter()
        for c in self.by_source.values():
            tot.update(c)
        return {
            "fingerprint": self.fp_meta,
            "rule": ("drop a record if any matcher fires on its ITEM-tier texts against any Index 0.2.1 row (all "
                     "catalogs, DEV and TEST, HLE included); whole-payload matches that are not item matches are "
                     "logged as context overlap and kept"),
            "totals": dict(tot),
            "by_source": {s: dict(c) for s, c in sorted(self.by_source.items())},
            "hits_by_source_and_index_catalog": {s: dict(sorted(c.items())) for s, c in sorted(self.pairs.items())},
            "dropped_examples": dict(self.examples),
            "context_overlap_examples": dict(self.ctx_examples),
        }
