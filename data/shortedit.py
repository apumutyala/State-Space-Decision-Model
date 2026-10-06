"""Short-item matchers (added after the PR #16 review): token-level edit distance and containment.

Why: items under ~13 words (BANKING77/CLINC utterances, When2Call questions, ANLI hypotheses) have no 13-gram,
and a one-word edit moves their character-5-gram Jaccard to ~0.55-0.62, below the MinHash threshold. The review
measured one-word-substitution recall of 13-17% on BANKING77/CLINC DEV items.

edit     a training segment of 4..32 tokens matches an indexed Index segment of 4..30 tokens when the token
         Levenshtein distance d satisfies d <= 2 AND d / max(len) <= 0.2 (the stricter of the two). Allowed d is
         therefore 0 below 5 tokens, 1 for 5..9, 2 from 10. Search is exact (no sampling): two sequences within
         distance d share a common variant obtained by deleting <= d tokens from each (substitution = one deletion
         on each side, insertion = one deletion on the longer side). Every Index item stores the hashes of its
         deletion variants with budget(L) = 0 (L < 5), 1 (5 <= L < 10), 2 (L >= 10) deletions, which covers every
         allowed pair (if the other side is longer, at least one edit is an insertion). Candidates are then verified
         with the exact distance.
contain  an indexed Index segment of 6..30 tokens occurs as a contiguous token run inside a training segment of
         <= 80 tokens (catches "item + appended sentence").

Not covered: truncations (a training item that is a fragment of an Index item), edits beyond the bound,
paraphrase, translation.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np

MIN_TOKENS = 4
MAX_TOKENS = 30
MAX_QUERY_TOKENS = MAX_TOKENS + 2
CONTAIN_MIN = 6
CONTAIN_MAX_QUERY = 80
MAX_EDITS = 2
MAX_RATIO = 0.2

_P = 1099511628211
_C = np.uint64(0x9E3779B97F4A7C15)
_M64 = (1 << 64) - 1


def budget(n: int) -> int:
    return 0 if n < 5 else 1 if n < 10 else 2


def allowed(n: int, m: int) -> int:
    """Largest d allowed between lengths n and m (stricter of <= 2 edits and <= 0.2 normalised)."""
    return min(MAX_EDITS, int(MAX_RATIO * max(n, m) + 1e-9))


@lru_cache(maxsize=None)
def _pow_matrix(L: int) -> np.ndarray:
    """(V, L) uint64: row v holds P^exponent for kept positions of deletion pattern v, 0 for deleted ones."""
    rows = []
    for k in range(budget(L) + 1):
        for dele in combinations(range(L), k):
            keep = [i for i in range(L) if i not in dele]
            r = [0] * L
            for rank, i in enumerate(keep):
                r[i] = pow(_P, len(keep) - 1 - rank, 1 << 64)
            rows.append(r)
    return np.array(rows, dtype=np.uint64)


@lru_cache(maxsize=None)
def _kept_len(L: int) -> np.ndarray:
    return np.array([L - k for k in range(budget(L) + 1) for _ in combinations(range(L), k)], dtype=np.uint64)


def variant_hashes(W: np.ndarray) -> np.ndarray:
    """W: (N, L) uint64 word ids of N sequences of equal length L -> (N, V) deletion-variant hashes."""
    L = W.shape[1]
    with np.errstate(over="ignore"):
        H = W @ _pow_matrix(L).T  # integer matmul wraps mod 2**64
        return H + _kept_len(L)[None, :] * _C


def run_hashes(ids: np.ndarray, k: int) -> np.ndarray:
    """Hashes of every contiguous run of k tokens, comparable with the 0-deletion variant hash."""
    m = ids.shape[0] - k + 1
    if m <= 0:
        return np.empty(0, dtype=np.uint64)
    h = np.zeros(m, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(k):
            h = h * np.uint64(_P) + ids[i:i + m]
        return h + np.uint64(k) * _C


def levenshtein(a, b, cap: int) -> int:
    """Token edit distance, early exit once it exceeds cap (returns cap + 1)."""
    n, m = len(a), len(b)
    if abs(n - m) > cap:
        return cap + 1
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        lo = cap + 1
        ai = a[i - 1]
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ai != b[j - 1]))
            lo = min(lo, cur[j])
        if lo > cap:
            return cap + 1
        prev = cur
    return prev[m]


class ShortIndex:
    """Index of short Index segments (MIN_TOKENS..MAX_TOKENS tokens) for the edit and contain matchers."""

    def __init__(self, tok: np.ndarray, off: np.ndarray, cat: np.ndarray):
        self.tok, self.off, self.cat = tok, off, cat
        n = off.shape[0] - 1
        lens = np.diff(off)
        keys, items, full_k, full_i = [], [], [], []
        for L in range(MIN_TOKENS, MAX_TOKENS + 1):
            idx = np.nonzero(lens == L)[0]
            if idx.size == 0:
                continue
            W = np.stack([tok[off[i]:off[i] + L] for i in idx]) if idx.size < 50 else \
                tok[(off[idx][:, None] + np.arange(L)[None, :])]
            H = variant_hashes(W)
            keys.append(H.ravel())
            items.append(np.repeat(idx.astype(np.int32), H.shape[1]))
            if L >= CONTAIN_MIN:
                full_k.append(H[:, 0])
                full_i.append(idx.astype(np.int32))
        k = np.concatenate(keys) if keys else np.empty(0, np.uint64)
        it = np.concatenate(items) if items else np.empty(0, np.int32)
        o = np.argsort(k, kind="stable")
        self.var_key, self.var_item = k[o], it[o]
        fk = np.concatenate(full_k) if full_k else np.empty(0, np.uint64)
        fi = np.concatenate(full_i) if full_i else np.empty(0, np.int32)
        o = np.argsort(fk, kind="stable")
        self.full_key, self.full_item = fk[o], fi[o]
        self.n_items = n

    @classmethod
    def from_token_lists(cls, toks: list[np.ndarray], cats: list[int]) -> "ShortIndex":
        lens = np.array([t.shape[0] for t in toks], dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        tok = np.concatenate(toks) if toks else np.empty(0, np.uint64)
        return cls(tok, off, np.array(cats, dtype=np.int32))

    def item(self, i: int) -> np.ndarray:
        return self.tok[self.off[i]:self.off[i + 1]]

    @staticmethod
    def _range(sorted_keys: np.ndarray, q: np.ndarray):
        lo = np.searchsorted(sorted_keys, q, side="left")
        hi = np.searchsorted(sorted_keys, q, side="right")
        return lo, hi

    def edit_hits(self, ids: np.ndarray) -> set[int]:
        """Catalogs of Index items within the allowed token edit distance of ``ids``."""
        n = ids.shape[0]
        if n < MIN_TOKENS or n > MAX_QUERY_TOKENS or self.var_key.size == 0:
            return set()
        q = variant_hashes(ids[None, :])[0]
        lo, hi = self._range(self.var_key, q)
        cand = set()
        for a, b in zip(lo[hi > lo], hi[hi > lo]):
            cand.update(self.var_item[a:b].tolist())
        out = set()
        if not cand:
            return out
        qa = ids.tolist()
        for i in cand:
            t = self.item(i)
            cap = allowed(n, t.shape[0])
            if levenshtein(qa, t.tolist(), cap) <= cap:
                out.add(int(self.cat[i]))
        return out

    def contain_hits(self, ids: np.ndarray) -> set[int]:
        """Catalogs of Index items (>= CONTAIN_MIN tokens) occurring as a contiguous run inside ``ids``."""
        n = ids.shape[0]
        if n < CONTAIN_MIN or n > CONTAIN_MAX_QUERY or self.full_key.size == 0:
            return set()
        out = set()
        for k in range(CONTAIN_MIN, min(MAX_TOKENS, n) + 1):
            q = run_hashes(ids, k)
            lo, hi = self._range(self.full_key, q)
            for a, b in zip(lo[hi > lo], hi[hi > lo]):
                for i in self.full_item[a:b].tolist():
                    t = self.item(i)
                    # verify (hash collision guard): the run must equal the item
                    s = ids.tolist()
                    tl = t.tolist()
                    if any(s[p:p + k] == tl for p in range(n - k + 1)):
                        out.add(int(self.cat[i]))
        return out
