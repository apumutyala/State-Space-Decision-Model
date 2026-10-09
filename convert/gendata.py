"""Q9-AF v3 data plumbing (jobs/q9af_v3): the long-context block split, teacher-generation prompts, completion packing
and the mixed (generic + teacher-generated) training stream. Pure numpy; the GPU parts are convert/sample.py and
convert/v3.py.

Every token comes from the SAME decontaminated mixture v2 builds (``convert.mixdata``, ``mix.decontam`` full-eval-v1:
DEV 13-grams + the pinned DEV/TEST/held-out eval-hash index + the eval attack templates). Nothing here reads an Index
row, a TEST / held-out / cohort-v3 row or any non-open-weight model output: prompts are prefixes of mixture documents,
and completions are sampled from the stock open-weight Qwen3.5-9B (the teacher) in phase G.

Block split (``BlockSplit``): train.u32 is cut into aligned blocks of ``lc_seq`` tokens (32,768). A seeded choice of
``n_lc`` blocks is reserved for the 32k long-context stage of job B (exported as ``lc.u32``); stage A's 4,096-token
windows and every generation prompt come only from the other blocks. Nothing is read twice across the two jobs.

Prompts (``select_prompts``): documents are the eos-terminated spans of train.u32 (sources in the stream order of
``mix.json``: web, chat, math). A chat doc's prompt is its tokens up to and including the first assistant header with
the empty think block (``<|im_start|>assistant\\n<think>\\n\\n</think>``, the readout's chat format); the teacher
writes the answer. A web / math doc's prompt is its first ``doc_prompt_tokens`` tokens; the teacher continues it.

Packing (``pack``): (prompt, completion) samples are laid end to end into rows of ``seq_len`` tokens with a 0/1 mask
that is 1 exactly on completion tokens (the loss is completion-only). A sample longer than a row is dropped (counted).
"""
from __future__ import annotations

import numpy as np

SOURCES = ("web", "chat", "math")
CHAT_ASSISTANT_HEADER = "<|im_start|>assistant\n<think>\n\n</think>"


# ----------------------------------------------------------------------------- block split
class BlockSplit:
    """Aligned ``lc_seq`` blocks of the train stream; ``n_lc`` of them (seeded) reserved for the long-context stage."""

    def __init__(self, n_tokens: int, seq_len: int, lc_seq: int, n_lc: int, seed: int):
        if lc_seq % seq_len:
            raise ValueError(f"lc_seq {lc_seq} must be a multiple of seq_len {seq_len}")
        self.n_tokens, self.seq_len, self.lc_seq, self.seed = int(n_tokens), seq_len, lc_seq, seed
        self.n_blocks = (self.n_tokens - 1) // lc_seq
        if n_lc > self.n_blocks // 4:
            raise ValueError(f"n_lc {n_lc} > a quarter of the {self.n_blocks} blocks")
        perm = np.random.default_rng(seed + 31).permutation(self.n_blocks)
        self.lc_blocks = np.sort(perm[:n_lc])
        self.is_lc = np.zeros(self.n_blocks + 1, dtype=bool)  # +1: the partial tail block is never LC
        self.is_lc[self.lc_blocks] = True
        self.per_block = lc_seq // seq_len

    def short_windows(self) -> np.ndarray:
        """Ids of the seq_len windows (window w = tokens [w*L, (w+1)*L)) outside every LC block."""
        n_win = (self.n_tokens - 1) // self.seq_len
        w = np.arange(n_win)
        return w[~self.is_lc[w // self.per_block]]

    def doc_ok(self, start: int, end: int) -> bool:
        """A doc [start, end) touches no LC block."""
        b0, b1 = start // self.lc_seq, (end - 1) // self.lc_seq
        return not self.is_lc[min(b0, self.n_blocks):min(b1, self.n_blocks) + 1].any()

    def lc_tokens(self, toks) -> np.ndarray:
        return np.concatenate([np.asarray(toks[b * self.lc_seq:(b + 1) * self.lc_seq], dtype=np.uint32)
                               for b in self.lc_blocks]) if len(self.lc_blocks) else np.zeros(0, np.uint32)

    def identity(self) -> dict:
        return {"lc_seq": self.lc_seq, "seq_len": self.seq_len, "n_blocks": int(self.n_blocks),
                "n_lc": int(len(self.lc_blocks)), "seed": self.seed, "perm_seed": self.seed + 31,
                "lc_tokens": int(len(self.lc_blocks) * self.lc_seq),
                "short_windows": int(len(self.short_windows()))}


# ----------------------------------------------------------------------------- documents and prompts
def doc_spans(toks, eos: int) -> np.ndarray:
    """[n_docs, 2] (start, end) of every eos-terminated doc (end includes the eos)."""
    t = np.asarray(toks)
    ends = np.flatnonzero(t == eos) + 1
    starts = np.concatenate([[0], ends[:-1]])
    return np.stack([starts, ends], 1).astype(np.int64)


def source_of(start: int, bounds: dict) -> str:
    for s in SOURCES:
        lo, hi = bounds[s]
        if lo <= start < hi:
            return s
    return "none"


def source_bounds(mix: dict) -> dict:
    """Token range of each source in train.u32 (mixdata writes web, chat, math in that order)."""
    out, off = {}, 0
    for s in SOURCES:
        n = int(mix["sources"][s]["tokens"])
        out[s] = (off, off + n)
        off += n
    return out


def find_sub(a: np.ndarray, pat: np.ndarray) -> int:
    """First index of ``pat`` in ``a`` (-1 if absent)."""
    n, m = len(a), len(pat)
    if m == 0 or n < m:
        return -1
    cand = np.flatnonzero(a[: n - m + 1] == pat[0])
    for i in cand:
        if np.array_equal(a[i:i + m], pat):
            return int(i)
    return -1


def select_prompts(toks, spans, bounds, split: BlockSplit, header_ids, counts: dict, cfg: dict, seed: int,
                   exclude: set | None = None):
    """Deterministic prompts: per source, docs in a seeded order, the first ``counts[src]`` that qualify.

    cfg: doc_prompt_tokens (web/math prompt length), min_continuation (web/math docs must be at least this much longer
    than the prompt), max_prompt_tokens (chat prompts longer than this are skipped). Returns (prompts, used doc ids,
    stats); each prompt = {"src", "doc", "ids": np.uint32}."""
    exclude = exclude or set()
    rng = np.random.default_rng(seed)
    header = np.asarray(header_ids, dtype=np.int64)
    by_src = {s: [] for s in SOURCES}
    for d, (a, b) in enumerate(spans):
        s = source_of(int(a), bounds)
        if s in by_src and d not in exclude:
            by_src[s].append(d)
    prompts, used, stats = [], set(), {}
    for s in SOURCES:
        order = rng.permutation(len(by_src[s])) if by_src[s] else []
        st = {"candidates": len(by_src[s]), "taken": 0, "skipped_lc": 0, "skipped_short": 0, "skipped_no_header": 0,
              "skipped_long": 0}
        for k in order:
            if st["taken"] >= counts.get(s, 0):
                break
            d = by_src[s][k]
            a, b = (int(x) for x in spans[d])
            if not split.doc_ok(a, b):
                st["skipped_lc"] += 1
                continue
            doc = np.asarray(toks[a:b], dtype=np.int64)
            if s == "chat":
                i = find_sub(doc, header)
                if i < 0:
                    st["skipped_no_header"] += 1
                    continue
                ids = doc[: i + len(header)]
                if len(ids) > cfg["max_prompt_tokens"]:
                    st["skipped_long"] += 1
                    continue
            else:
                n = cfg["doc_prompt_tokens"]
                if len(doc) < n + cfg["min_continuation"]:
                    st["skipped_short"] += 1
                    continue
                ids = doc[:n]
            prompts.append({"src": s, "doc": d, "ids": ids.astype(np.uint32)})
            used.add(d)
            st["taken"] += 1
        stats[s] = st
    return prompts, used, stats


def split_counts(n: int, shares: dict) -> dict:
    """Largest-remainder split of n over the sources."""
    raw = {s: n * shares.get(s, 0.0) for s in SOURCES}
    out = {s: int(np.floor(v)) for s, v in raw.items()}
    for s in sorted(SOURCES, key=lambda s: -(raw[s] - out[s]))[: n - sum(out.values())]:
        out[s] += 1
    return out


def save_prompts(path, prompts) -> None:
    lens = np.asarray([len(p["ids"]) for p in prompts], dtype=np.int64)
    np.savez(path, toks=np.concatenate([p["ids"] for p in prompts]) if prompts else np.zeros(0, np.uint32),
             lens=lens, src=np.asarray([SOURCES.index(p["src"]) for p in prompts], dtype=np.int64),
             doc=np.asarray([p["doc"] for p in prompts], dtype=np.int64))


def load_prompts(path) -> list:
    z = np.load(path, allow_pickle=False)
    offs = np.concatenate([[0], np.cumsum(z["lens"])])
    return [{"src": SOURCES[int(s)], "doc": int(d), "ids": z["toks"][offs[i]:offs[i + 1]]}
            for i, (s, d) in enumerate(zip(z["src"], z["doc"]))]


# ----------------------------------------------------------------------------- packing
def pack(samples, seq_len: int, pad_id: int):
    """samples: iterable of (prompt ids, completion ids). Returns (rows uint32 [R, L], mask uint8 [R, L], stats).
    mask[r, t] = 1 iff token t of row r is a completion token. Rows are filled left to right; a sample that does not
    fit the rest of the row starts a new row (the rest is padded with ``pad_id``, mask 0)."""
    rows, masks, cur, cm = [], [], [], []
    st = {"samples": 0, "dropped_too_long": 0, "dropped_empty": 0, "completion_tokens": 0, "prompt_tokens": 0}

    def flush():
        if cur:
            r = np.full(seq_len, pad_id, dtype=np.uint32)
            m = np.zeros(seq_len, dtype=np.uint8)
            r[: len(cur)] = cur
            m[: len(cm)] = cm
            rows.append(r)
            masks.append(m)
            cur.clear()
            cm.clear()

    for p, c in samples:
        p, c = list(map(int, p)), list(map(int, c))
        if not c:
            st["dropped_empty"] += 1
            continue
        n = len(p) + len(c)
        if n > seq_len:
            st["dropped_too_long"] += 1
            continue
        if len(cur) + n > seq_len:
            flush()
        cur.extend(p + c)
        cm.extend([0] * len(p) + [1] * len(c))
        st["samples"] += 1
        st["completion_tokens"] += len(c)
        st["prompt_tokens"] += len(p)
    flush()
    R = np.stack(rows) if rows else np.zeros((0, seq_len), np.uint32)
    M = np.stack(masks) if masks else np.zeros((0, seq_len), np.uint8)
    st["rows"] = int(len(R))
    st["fill"] = float(M.size and (st["completion_tokens"] + st["prompt_tokens"]) / M.size)
    return R, M, st


def pred_weights(mask: np.ndarray) -> np.ndarray:
    """Prediction-position weights from a token mask: position t predicts token t+1, so w[:, t] = mask[:, t+1]
    (the last position predicts nothing in the row: 0)."""
    w = np.zeros(mask.shape, dtype=np.float32)
    w[:, :-1] = mask[:, 1:]
    return w


# ----------------------------------------------------------------------------- streams
class IndexedWindowStream:
    """WindowStream over an explicit list of window ids (the non-LC windows): one seeded permutation, each window at
    most once per pass; a new pass is a fresh permutation and is counted (reported), never a crash."""

    def __init__(self, toks, seq_len, batch, window_ids, seed, device="cpu"):
        self.t, self.L, self.B, self.ids, self.seed, self.device = toks, seq_len, batch, np.asarray(window_ids), seed, device
        self.passes, self.used = 0, 0
        self._perm()

    def _perm(self):
        self.order = self.ids[np.random.default_rng(self.seed + self.passes).permutation(len(self.ids))]
        self.pos = 0

    @property
    def windows_used(self):
        return self.used

    def take(self, b=None):
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

        idx = self.take()
        x = np.stack([np.asarray(self.t[i * self.L:(i + 1) * self.L], dtype=np.int64) for i in idx])
        return torch.from_numpy(x).to(self.device)

    def __iter__(self):
        return self


class MixedStream:
    """Batches of either generic windows (full-sequence KL) or teacher-generated rows (completion-only loss).

    Step s is a generated batch iff floor((s+1) f) > floor(s f) (an exact ``gen_frac`` f schedule) and unread
    generated rows remain; each generated row is read once, in one seeded order. Yields (x, w, kind): w is None for a
    generic batch, else the [B, L] float prediction weights (``pred_weights`` of the completion mask)."""

    def __init__(self, generic, gen_rows, gen_mask, batch, gen_frac, seed, device="cpu"):
        self.g, self.R, self.M, self.B, self.f, self.device = generic, gen_rows, gen_mask, batch, gen_frac, device
        self.order = np.random.default_rng(seed + 53).permutation(len(gen_rows)) if len(gen_rows) else np.zeros(0, int)
        self.pos, self.step, self.n_gen, self.n_generic = 0, 0, 0, 0

    @property
    def gen_left(self):
        return len(self.order) - self.pos

    def __next__(self):
        import torch

        s, f = self.step, self.f
        self.step += 1
        if np.floor((s + 1) * f) > np.floor(s * f) and self.gen_left >= self.B:
            idx = self.order[self.pos:self.pos + self.B]
            self.pos += self.B
            self.n_gen += 1
            x = torch.from_numpy(self.R[idx].astype(np.int64)).to(self.device)
            w = torch.from_numpy(pred_weights(self.M[idx])).to(self.device)
            return x, w, "gen"
        self.n_generic += 1
        return next(self.g), None, "generic"

    def __iter__(self):
        return self

    def state(self) -> dict:
        return {"steps": self.step, "gen_batches": self.n_gen, "generic_batches": self.n_generic,
                "gen_rows_used": int(self.pos), "gen_rows_total": int(len(self.order)),
                "generic_windows_used": int(self.g.windows_used), "generic_passes": int(getattr(self.g, "passes", 0))}


# ----------------------------------------------------------------------------- recall-style teacher prompts
#: Hand-written, generic instructions (no eval item, no benchmark wording). Phase G checks every one against the
#: shipped full-eval-v1 filter (eval-hash index + attack templates) and refuses to run if any is flagged.
INSTRUCTIONS = {
    "retrieval": [
        "List every number that appears in the text above, each with what it refers to.",
        "Which people, places or organisations does the text above mention? Quote the sentence where each first appears.",
        "Copy the second sentence of the text above exactly, word for word.",
        "What does the text above say right after the words \"{span}\"? Quote it exactly.",
    ],
    "tool": [
        "You can call one tool, search(query: str). Reply with the single JSON call that would best find more "
        "information about the main topic of the text above.",
        "You have two tools: extract_dates(text: str) and extract_entities(text: str). Reply with a JSON list of the "
        "calls you would make, with arguments copied exactly from the text above.",
    ],
    "knowledge": [
        "Summarise the key facts of the text above in three bullet points, keeping every name and number exact.",
        "Write two questions that the text above answers, each followed by its answer quoted from the text.",
    ],
}
INSTRUCTION_KINDS = tuple(INSTRUCTIONS)


def instruction_prompts(tok, toks, spans, bounds, split, counts: dict, cfg: dict, seed: int, exclude: set | None = None,
                        system: str = "You are a helpful assistant."):
    """Chat prompts that ask the teacher to RETRIEVE from a mixture document (web / math, <= ``doc_tokens`` tokens):
    ``<|im_start|>user\n<doc>\n\n<instruction><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>``.
    Returns (prompts, used docs, stats); prompt["src"] is the kind (retrieval / tool / knowledge)."""
    exclude = set(exclude or ())
    rng = np.random.default_rng(seed)
    cand = [d for d, (a, _) in enumerate(spans) if source_of(int(a), bounds) in ("web", "math") and d not in exclude]
    order = rng.permutation(len(cand)) if cand else []
    prompts, used, st = [], set(), {k: {"taken": 0} for k in INSTRUCTIONS}
    want = dict(counts)
    k_iter = iter(order)
    for kind in INSTRUCTIONS:
        while st[kind]["taken"] < want.get(kind, 0):
            try:
                d = cand[next(k_iter)]
            except StopIteration:
                break
            a, b = (int(x) for x in spans[d])
            if not split.doc_ok(a, b) or b - a < cfg["min_doc_tokens"]:
                continue
            ids = np.asarray(toks[a:min(b - 1, a + cfg["doc_tokens"])], dtype=np.int64)
            text = tok.decode(ids.tolist())
            instr = INSTRUCTIONS[kind][int(rng.integers(len(INSTRUCTIONS[kind])))]
            if "{span}" in instr:
                p = int(rng.integers(0, max(1, len(ids) - 8)))
                instr = instr.replace("{span}", tok.decode(ids[p:p + 5].tolist()).strip())
            chat = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{text}\n\n{instr}<|im_end|>\n"
                    "<|im_start|>assistant\n<think>\n\n</think>")
            pid = np.asarray(tok(chat, add_special_tokens=False)["input_ids"], dtype=np.uint32)
            if len(pid) > cfg["max_prompt_tokens"]:
                continue
            prompts.append({"src": kind, "doc": d, "ids": pid})
            used.add(d)
            st[kind]["taken"] += 1
    return prompts, used, st


def check_templates(doc_filter) -> dict:
    """full-eval-v1 flags of every instruction template (doc_filter = convert.mixdata.full_filter()[0])."""
    texts = [t for v in INSTRUCTIONS.values() for t in v]
    flags = doc_filter(texts)
    return {"templates": len(texts), "flagged": [t for t, f in zip(texts, flags) if f]}
