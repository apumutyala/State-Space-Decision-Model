"""G4 erase-and-check with exact recurrent-state reuse (snapshot / prefix-trie).

The v0.1 certificate (``safety_eval/certify.py``) recomputes every origin from scratch: 1 + sum_{j<=k} C(n, j)
full forward passes. A recurrent backbone's state after a token prefix is a deterministic function of that
prefix, so every origin that shares a token prefix with another can resume from the shared state.

``TrieEvaluator`` builds the token trie of all origin sequences (full request + every erasure origin), walks
it depth-first, runs each edge's tokens once, and clones the state only at branch points. It is exact in the
only sense that matters: the computation for each origin is the same recurrence applied to the same tokens,
resumed from a state produced by the same tokens (float caveat: a chunked kernel re-chunked at a resume point
may differ in the last bits; decisions are compared to a full recompute and ``max_abs_logit_diff`` is reported).
No tokenizer compositionality is assumed: origins are tokenised exactly as the decide path would tokenise them
(``row_builder``), and the trie shares only literally identical token prefixes.

Memory: the DFS stack holds one state per open branch point; for k = 1 that is O(n) states (one per sentence
boundary of the untrusted field), i.e. "store state at every sentence boundary".

Scope: decisions that are a function of the FINAL feature (plus optional option continuations, the fork
head). Heads that integrate per-token features over the evidence span would also need the cached prefix
features (equal by causality); not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch

from .adapters import Runner

ReadoutFn = Callable[[torch.Tensor], torch.Tensor]  # final feature [d] -> logits [K]


def random_readout(d: int, k: int, seed: int = 0) -> ReadoutFn:
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(k, d, generator=g, dtype=torch.float64)

    def f(h: torch.Tensor) -> torch.Tensor:
        return W.to(h.dtype) @ h
    return f


@dataclass
class TrieResult:
    logits: list[torch.Tensor]
    tokens: int  # tokens run through the backbone
    calls: int  # backbone forward calls (edge runs)
    max_states_held: int
    extra: dict[str, Any] = field(default_factory=dict)


class TrieEvaluator:
    def __init__(self, runner: Runner, readout: ReadoutFn, conts: Sequence[Sequence[int]] | None = None) -> None:
        """``conts``: optional option continuations (fork head): logits[k] = readout(feature after cont_k)[0]
        is replaced by the scalar ``readout(h_k).sum()``, i.e. a per-option score."""
        self.runner, self.readout, self.conts = runner, readout, conts

    def _decide(self, state, h):
        if not self.conts:
            return self.readout(h)
        scores = []
        for c in self.conts:
            _, hk = self.runner.run(c, self.runner.clone(state))
            scores.append(self.readout(hk).sum())
        return torch.stack(scores)

    def evaluate(self, seqs: Sequence[Sequence[int]]) -> TrieResult:
        seqs = [list(s) for s in seqs]
        if any(len(s) == 0 for s in seqs):
            raise ValueError("empty sequence")
        r0 = (self.runner.tokens, self.runner.calls)
        out: list[torch.Tensor | None] = [None] * len(seqs)
        held = [0, 0]

        def node(group: list[int], depth: int, state, h):
            # extend the common prefix of the whole group
            end = min(len(seqs[i]) for i in group)
            lcp = depth
            while lcp < end and all(seqs[i][lcp] == seqs[group[0]][lcp] for i in group):
                lcp += 1
            if lcp > depth:
                state, h = self.runner.run(seqs[group[0]][depth:lcp], state)
            done = [i for i in group if len(seqs[i]) == lcp]
            rest = [i for i in group if len(seqs[i]) > lcp]
            children: dict[int, list[int]] = {}
            for i in rest:
                children.setdefault(seqs[i][lcp], []).append(i)
            parts = list(children.values())
            n_uses = len(parts) + (1 if done else 0)
            if n_uses > 1:
                held[0] += 1
                held[1] = max(held[1], held[0])
            if done:
                d = self._decide(state, h)  # never mutates ``state`` (continuations run on clones)
                for i in done:
                    out[i] = d
            for j, part in enumerate(parts):
                last = j == len(parts) - 1
                node(part, lcp, state if last else self.runner.clone(state), h)
            if n_uses > 1:
                held[0] -= 1

        node(list(range(len(seqs))), 0, None, None)
        return TrieResult(out, self.runner.tokens - r0[0], self.runner.calls - r0[1], held[1])

    def full_recompute(self, seqs: Sequence[Sequence[int]]) -> TrieResult:
        r0 = (self.runner.tokens, self.runner.calls)
        out = []
        for s in seqs:
            st, h = self.runner.run(s, None)
            out.append(self._decide(st, h))
        return TrieResult(out, self.runner.tokens - r0[0], self.runner.calls - r0[1], 0)


# ----------------------------------------------------------------------------------- token-level erase-and-check
def erasure_origins(prefix: Sequence[int], segs: Sequence[Sequence[int]], suffix: Sequence[int], k: int):
    """(erased set, token sequence) for the full sequence (erased = ()) and every erasure of 1..k segments."""
    import itertools

    out = [((), list(prefix) + [t for s in segs for t in s] + list(suffix))]
    for j in range(1, min(k, len(segs)) + 1):
        for E in itertools.combinations(range(len(segs)), j):
            out.append((E, list(prefix) + [t for i, s in enumerate(segs) if i not in E for t in s] + list(suffix)))
    return out


def snapshot_certify(runner: Runner, readout: ReadoutFn, origins: Sequence[tuple[Any, Sequence[int]]],
                     conts: Sequence[Sequence[int]] | None = None, verify: bool = False) -> dict[str, Any]:
    """Certificate over ``origins`` (first entry = the observed request). ``certified`` iff every origin's
    argmax equals the observed one. With ``verify`` also runs the C(n,k) full-recompute baseline and checks
    that every origin's decision is identical (``decisions_identical``)."""
    ev = TrieEvaluator(runner, readout, conts)
    seqs = [s for _, s in origins]
    res = ev.evaluate(seqs)
    dec = [int(torch.argmax(l)) for l in res.logits]
    rec: dict[str, Any] = {
        "schema": "ssdm.g4.snapshot.v1", "n_origins": len(seqs) - 1, "decision": dec[0],
        "certified": all(d == dec[0] for d in dec), "changed": [list(origins[i][0]) for i, d in enumerate(dec) if d != dec[0]],
        "snapshot_tokens": res.tokens, "snapshot_calls": res.calls, "max_states_held": res.max_states_held,
        # baseline = one fresh pass per origin, plus the same option continuations per origin
        "baseline_tokens": sum(len(s) for s in seqs) + len(seqs) * sum(len(c) for c in (conts or [])),
        "baseline_calls": len(seqs) * (1 + len(conts or [])),
    }
    rec["token_savings"] = 1.0 - rec["snapshot_tokens"] / rec["baseline_tokens"]
    rec["call_ratio"] = rec["snapshot_calls"] / rec["baseline_calls"]  # >1 is common: more, shorter runs
    rec["forward_equivalents_snapshot"] = rec["snapshot_tokens"] / len(seqs[0])
    rec["forward_equivalents_baseline"] = rec["baseline_tokens"] / len(seqs[0])
    if verify:
        base = ev.full_recompute(seqs)
        bdec = [int(torch.argmax(l)) for l in base.logits]
        rec["decisions_identical"] = bdec == dec
        rec["max_abs_logit_diff"] = float(max((a - b).abs().max() for a, b in zip(res.logits, base.logits)))
        rec["baseline_tokens_measured"] = base.tokens
    return rec


def request_origins(request: dict, k: int, row_ids: Callable[[dict], Sequence[int]]) -> list[tuple[Any, list[int]]]:
    """Text-level bridge to ``safety_eval.certify``: the observed request plus its covered-insertion origins
    (same segmenter, whitespace variants and hash dedup as ``certify.certify``), each tokenised by the real
    row builder ``row_ids(request) -> ids`` (e.g. ``heads.layout.build_row(...).ids``)."""
    from safety_eval import certify as C

    out = [((), list(row_ids(request)))]
    seen = {C.request_hash(request)}
    for lab, r in C.origin_requests(request, k):
        h = C.request_hash(r)
        if h in seen:
            continue
        seen.add(h)
        out.append((tuple(lab), list(row_ids(r))))
    return out
