"""Q9-AF v3 recall-heavy training rows (jobs/q9af_v3/README.md "recall"): synthetic associative-recall / needle /
quote-continuation tasks built from the decontaminated mixture's OWN tokens, with completion-only answer masks.

Why: v2 (job-7ji99) kept held-out KL at 0.097 but lost recall: MQAR drop 0.59, NIAH drop 0.525, and proxy600
retrieval 36.9 -> 8.1. Generic KD barely exercises exact retrieval over long contexts. These rows do, and only that.

Tasks (each row = a prefix of real mixture text + one task, filled to seq_len; the loss is on the answers only):
  kv      an inventory of N key -> id records (N in kv_pairs), then Q lookups "Q: id of item <key>? A: <id>"
  needle  1-4 memos "[memo] The access number for <name> is <n>." at random depths of real text, then a recall
          question per memo
  quote   real text, then "Continue the quote: <6 tokens from the text> ..." -> the next 12 tokens (extractive recall)
Keys are word-like tokens drawn from the mixture stream. The recall PROBE's key words (``probes.core._key_words``),
its templates ("Memorize these key-value pairs", "The secret passkey is"), its value ranges and its filler (wikitext
eval) are never used, so the gate stays held-out in format. No eval item is involved anywhere: the text is the
full-eval-v1 decontaminated mix, and the templates are the fixed strings below.
"""
from __future__ import annotations

import numpy as np

KV_HEAD = "Inventory log:\n"
KV_REC = " item{k} has id {v}\n"
KV_Q = "Q: id of item{k}? A:"
NEEDLE = " [memo] The access number for{k} is {v}. "
NEEDLE_Q = "\nRecall: the access number for{k} is"
QUOTE_Q = "\nContinue the quote:"
QUOTE_SEP = " ..."
TASKS = ("kv", "needle", "quote")


class RecallSynth:
    """Deterministic, unbounded stream of recall rows. ``rows(n)`` -> (x int64 [n, L], w float32 [n, L]) where w are
    PREDICTION-position weights (position t predicts token t + 1), 1 exactly where the next token is an answer token."""

    def __init__(self, tok, toks, window_ids, seq_len, seed, cfg, exclude_words=()):
        self.tok, self.t, self.L, self.seed, self.cfg = tok, toks, seq_len, seed, cfg
        self.wins = np.asarray(window_ids)
        self.calls = 0
        rng = np.random.default_rng(seed + 97)
        enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
        self.enc = enc
        excl = {w.strip().lower() for w in exclude_words}
        pool, seen = [], set()
        n = len(toks)
        for p in rng.integers(0, max(1, n - 1), size=cfg.get("key_scan", 200000)):
            i = int(toks[p])
            if i in seen:
                continue
            seen.add(i)
            s = tok.decode([i])
            w = s.strip()
            if cfg.get("key_filter", "word") == "any":  # tests' word-level tokenizer: any non-special, non-excluded id
                ok = bool(w) and w.lower() not in excl and i not in set(getattr(tok, "all_special_ids", []))
            else:  # a BPE word piece: " word" (leading space), alphabetic, >= 3 letters
                ok = s.startswith(" ") and w.isalpha() and len(w) >= 3 and w.lower() not in excl
            if ok:
                pool.append(i)
            if len(pool) >= cfg.get("key_pool", 4000):
                break
        if len(pool) < cfg.get("min_key_pool", 64):
            raise ValueError(f"recall key pool too small ({len(pool)} word-like tokens)")
        self.keys = np.asarray(pool, dtype=np.int64)
        self.head = enc(KV_HEAD)
        self.pieces = {k: [enc(x) for x in v.split("{k}")] for k, v in
                       (("rec", KV_REC.split("{v}")[0]), ("q", KV_Q), ("needle", NEEDLE.split("{v}")[0]),
                        ("needle_q", NEEDLE_Q))}
        self.rec_tail = enc(KV_REC.split("{v}")[1])
        self.needle_tail = enc(NEEDLE.split("{v}")[1])
        self.nl = enc("\n")
        self.quote_q, self.quote_sep = enc(QUOTE_Q), enc(QUOTE_SEP)

    def _val(self, rng):
        return self.enc(f" {int(rng.integers(1000, 1000000))}")  # the probe uses 100-999 and 5-digit passkeys

    def _filler(self, rng, n):
        out = []
        while len(out) < n:
            w = int(self.wins[rng.integers(len(self.wins))])
            out += np.asarray(self.t[w * self.L:(w + 1) * self.L], dtype=np.int64).tolist()
        return out[:n]

    def _key(self, rng, k=2):
        return self.keys[rng.choice(len(self.keys), size=k, replace=False)].tolist()

    def _kv(self, rng, N=None):
        N = N or int(rng.choice(self.cfg["kv_pairs"]))
        keys = [self._key(rng) for _ in range(N)]
        vals = [self._val(rng) for _ in range(N)]
        a, b = self.pieces["rec"]
        body = list(self.head)
        for k, v in zip(keys, vals):
            body += a + k + b + v + self.rec_tail
        ans = []
        qa, qb = self.pieces["q"]
        for qi in rng.choice(N, size=min(self.cfg["kv_queries"], N), replace=False):
            body += qa + keys[qi] + qb
            ans.append((len(body), len(vals[qi])))
            body += vals[qi] + self.nl
        return body, ans

    def _needle(self, rng, budget, m=None):
        m = m or int(rng.integers(1, 5))
        names = [self._key(rng) for _ in range(m)]
        vals = [self._val(rng) for _ in range(m)]
        a, b = self.pieces["needle"]
        needles = [a + k + b + v + self.needle_tail for k, v in zip(names, vals)]
        qa, qb = self.pieces["needle_q"]
        q_len = sum(len(qa) + len(k) + len(qb) + len(v) for k, v in zip(names, vals))
        n_fill = max(16, budget - sum(map(len, needles)) - q_len)
        fill = self._filler(rng, n_fill)
        at = sorted(int(x) for x in rng.integers(0, n_fill, size=m))
        body, last = [], 0
        for pos, nd in zip(at, needles):
            body += fill[last:pos] + nd
            last = pos
        body += fill[last:]
        ans = []
        for qi in rng.permutation(m):
            body += qa + names[qi] + qb
            ans.append((len(body), len(vals[qi])))
            body += vals[qi]
        return body, ans

    def _quote(self, rng, budget, nq=None):
        nq = nq or int(self.cfg.get("quote_questions", 4))
        n_fill = max(24, budget - nq * (len(self.quote_q) + 6 + len(self.quote_sep) + 12))
        fill = self._filler(rng, n_fill)
        body, ans = list(fill), []
        for p in rng.integers(0, max(1, n_fill - 18), size=nq):
            p = int(p)
            body += self.quote_q + fill[p:p + 6] + self.quote_sep
            ans.append((len(body), 12))
            body += fill[p + 6:p + 18]
        return body, ans

    def _row(self, rng):
        task = TASKS[int(rng.choice(len(TASKS), p=self.cfg["task_shares"]))]
        budget = int(self.L * rng.uniform(0.25, 1.0))
        if task == "kv":
            body, ans = self._kv(rng)
            N = int(self.cfg["kv_pairs"][0])
            while len(body) > self.L and N >= 1:  # never cut a record its question asks about: shrink instead
                body, ans = self._kv(rng, N)
                N //= 2
        elif task == "needle":
            body, ans = self._needle(rng, budget)
            if len(body) > self.L:
                body, ans = self._needle(rng, self.L // 2, m=1)
        else:
            body, ans = self._quote(rng, budget)
            if len(body) > self.L:
                body, ans = self._quote(rng, self.L // 2, nq=1)
        if len(body) > self.L:  # the task cannot fit this row length at all: a plain text row, no answer weight
            body, ans, task = [], [], "none"
        pre = self.L - len(body)
        x = np.asarray(self._filler(rng, pre) + body, dtype=np.int64)
        w = np.zeros(self.L, dtype=np.float32)
        for s, n in ans:
            w[pre + s - 1:pre + s - 1 + n] = 1.0  # positions predicting the answer tokens
        return x, w, task

    def rows(self, n):
        rng = np.random.default_rng([self.seed, self.calls])
        self.calls += 1
        xs, ws = zip(*[self._row(rng)[:2] for _ in range(n)])
        return np.stack(xs), np.stack(ws)


class MultiStream:
    """Batches from several sources with pre-registered fractions: ``generic`` (an IndexedWindowStream: full-sequence
    KL), ``gen`` (teacher-generated rows + mask, each read once: completion-only KL + CE), ``recall`` (RecallSynth:
    answer-only KL + CE). Smooth weighted round robin: every step each LIVE source earns its fraction in credit, the
    source with the most credit is taken and pays 1 (ties in the listed order), so any window of steps holds the
    fractions to within one batch. An exhausted ``gen`` hands its share to ``generic`` from then on (recall keeps its
    share). Yields (x, w | None, kind)."""

    def __init__(self, generic, fracs, batch, seed, gen_rows=None, gen_mask=None, recall=None, device="cpu"):
        self.g, self.f, self.B, self.device = generic, dict(fracs), batch, device
        if abs(sum(self.f.values()) - 1) > 1e-9:
            raise ValueError(f"fractions {self.f} do not sum to 1")
        self.R, self.M, self.rec = gen_rows, gen_mask, recall
        n = 0 if gen_rows is None else len(gen_rows)
        self.order = np.random.default_rng(seed + 53).permutation(n)
        self.pos = 0
        self.taken = {k: 0 for k in self.f}
        self.credit = {k: 0.0 for k in self.f}
        self.step = 0

    def _pick(self):
        live = [k for k in self.f if not (k == "gen" and len(self.order) - self.pos < self.B)
                and not (k == "recall" and self.rec is None)]
        frac = {k: self.f[k] for k in live}
        lost = sum(v for k, v in self.f.items() if k not in live)
        frac["generic"] = frac.get("generic", 0.0) + lost
        for k, v in frac.items():
            self.credit[k] += v
        k = max(frac, key=lambda k: (self.credit[k], -list(self.f).index(k)))
        self.credit[k] -= 1.0
        return k

    def __next__(self):
        import torch

        from convert.gendata import pred_weights

        k = self._pick()
        self.step += 1
        self.taken[k] += 1
        if k == "gen":
            idx = self.order[self.pos:self.pos + self.B]
            self.pos += self.B
            x = torch.from_numpy(np.asarray(self.R[idx]).astype(np.int64)).to(self.device)
            return x, torch.from_numpy(pred_weights(np.asarray(self.M[idx]))).to(self.device), "gen"
        if k == "recall":
            x, w = self.rec.rows(self.B)
            return torch.from_numpy(x).to(self.device), torch.from_numpy(w).to(self.device), "recall"
        return next(self.g), None, "generic"

    def __iter__(self):
        return self

    def state(self) -> dict:
        return {"steps": self.step, "taken": dict(self.taken), "credit": dict(self.credit), "gen_rows_used": int(self.pos),
                "gen_rows_total": int(len(self.order)), "recall_calls": int(self.rec.calls if self.rec else 0),
                "generic_windows_used": int(self.g.windows_used), "generic_passes": int(getattr(self.g, "passes", 0))}

    def restore(self, st: dict) -> None:
        """Fast-forward a fresh stream (same seeds) to a saved ``state()`` without reading any data twice."""
        self.step, self.taken, self.pos = st["steps"], dict(st["taken"]), st["gen_rows_used"]
        self.credit = dict(st["credit"])
        if self.rec is not None:
            self.rec.calls = st["recall_calls"]
        while self.g.windows_used < st["generic_windows_used"]:
            self.g.take()
