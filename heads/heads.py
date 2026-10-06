"""Decision heads for the m2 factorial (PLAN.md D-architecture).

Every head maps a ``HeadInput`` to per-option logits ``[B, K]`` (``-inf`` on padded
options) and shares parameters across the option axis, so it is permutation-
equivariant in the options given equivariant inputs (tests check this numerically;
float reduction order is the only source of difference). Typed outputs (noul,
choice, score) are applied on top by ``heads.typed``.

Novelty, stated as in ``analysis/B-claude-plan-review.md`` §1 row 8: these are known
components in a new application. The integrator is unnormalised linear attention
with the option as the query (Katharopoulos 2020; AFT 2021) with SENN/NAM-style
additive accounting; the snapshot head is chunk checkpointing (Marconi) plus
bounded-slot memories (ABC, GSA) without the softmax; Head B is Codex's
per-option selective recurrence (plans/partner/PLAN.md "Two heads to compare").

Attribution is ACCOUNTING-exact, not causal (B-review §3.5, CODEX_CRITICAL_COMPARISON):
the increment booked at token t is computed from a backbone feature that already
mixes all earlier tokens, so it says where evidence was booked, not which token
caused it. Causal claims need perturb-and-recompute (``substitute_token_effect``).

Heads
-----
(a) ``integrator``  z_k = b(e_k, q) + sum_t m_t * delta(h_t, e_k, q). lambda = 1 (no decay,
    no normalisation). Returns per-token attribution that sums to the logit exactly.
(b) ``fork``        z_k = g(f_k), f_k = backbone feature after prefix + continuation_k.
(c) ``snapshot``    evidence split into at most S chunks (S fixed, so head memory is
    O(S*K) scalars regardless of length); per-chunk partial sums of the integrator
    increments plus a chunk-boundary readout, aggregated by sum / max / gated unnormalised
    sum (an independent sigmoid per slot, no normalisation across slots). No softmax over
    positions or chunks; no backbone state is snapshotted. Caveats: equal-count chunking needs
    the evidence length before the scan (fine with options-first rows, whose length is known;
    not a streaming construct), and a streaming implementation must reduce each boundary
    readout (S x D) to its S x K scores at the boundary to keep the O(S*K) claim.
(d) ``head_b``      r[t,k] = a[t,k] * r[t-1,k] + b[t,k] * v[t]; (a, b) = f(v_t, e_k, q, s);
    z_k = g(r[T,k], e_k, s), s = symmetric mean summary of options. O(KTd) work, O(Kd) state.
(e) ``clef_control`` Clef-SHAPED control: options cross-attend over the whole sequence
    (2 routing layers) then self-attend among options (4 layers), with the released head
    config sizes. Shape-matched, not recipe-matched: Clef's scorer is a prior + gated joint
    over lexical/contextual option spans with state-first ordering (reviews/01), ~128M params
    vs ~89M here. Uses softmax over positions: a priced CONTROL, never the entrant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .features import HeadInput

NEG = float("-inf")


@dataclass
class HeadOutput:
    logits: torch.Tensor  # [B, K], -inf on padded options
    bias: torch.Tensor | None = None  # [B, K] evidence-free part (prior), integrator/snapshot
    attribution: torch.Tensor | None = None  # [B, T, K] (integrator) or [B, S, K] (snapshot)


def _masked(z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return z.masked_fill(~mask, NEG)


def _sym_summary(phi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(phi.dtype)
    return (phi * m).sum(1) / m.sum(1).clamp_min(1)


class DecisionHead(nn.Module):
    name = "base"
    attention_free = True
    needs_fork = False

    def cost(self, T: int, K: int) -> dict:
        """Per-request head cost beyond the backbone (FLOPs ~ 2 * MACs), and persistent state."""
        raise NotImplementedError

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------- (a) integrator bank
class IntegratorBank(DecisionHead):
    name = "integrator"

    def __init__(self, d_model: int, channels: int = 64, t_chunk: int = 256) -> None:
        super().__init__()
        self.C, self.t_chunk = channels, t_chunk
        self.u = nn.Linear(d_model, channels)
        self.v = nn.Linear(d_model, channels)
        self.vq = nn.Linear(d_model, channels, bias=False)
        self.gate = nn.Linear(d_model, 1)
        self.w = nn.Parameter(torch.randn(channels) / math.sqrt(channels))
        self.prior = nn.Sequential(nn.Linear(2 * d_model, channels), nn.SiLU(), nn.Linear(channels, 1))
        nn.init.constant_(self.gate.bias, -2.0)  # sparse booking at init

    def increments(self, x: HeadInput, t0: int, t1: int) -> torch.Tensor:
        """delta[b, t, k] for t in [t0, t1): zero outside the evidence region."""
        h = x.h[:, t0:t1]
        g = torch.sigmoid(self.gate(h)).squeeze(-1) * x.evidence_mask[:, t0:t1]
        uo = self.u(h)  # [B, t, C]
        ve = self.v(x.e) + self.vq(x.q_pre).unsqueeze(1)  # [B, K, C]
        inter = F.silu(uo.unsqueeze(2) * ve.unsqueeze(1))  # [B, t, K, C]
        return g.unsqueeze(-1) * (inter @ self.w) / math.sqrt(self.C)

    def forward(self, x: HeadInput, return_attribution: bool = False) -> HeadOutput:
        bsz, t, _ = x.h.shape
        k = x.e.shape[1]
        qe = x.q_pre.unsqueeze(1).expand(-1, k, -1)
        bias = self.prior(torch.cat([x.e, qe], -1)).squeeze(-1)
        acc = x.h.new_zeros((bsz, k))
        attr = [] if return_attribution else None
        for t0 in range(0, t, self.t_chunk):
            d = self.increments(x, t0, min(t, t0 + self.t_chunk))
            acc = acc + d.sum(1)
            if attr is not None:
                attr.append(d)
        logits = _masked(bias + acc, x.option_mask)
        return HeadOutput(logits, bias, torch.cat(attr, 1) if attr is not None else None)

    def cost(self, T: int, K: int) -> dict:
        return {"flops": 2 * T * (self.u.in_features * self.C) + 2 * T * K * 2 * self.C,
                "state_scalars": K, "attention_free": True}


# ---------------------------------------------------------------- (b) final-state fork
class FinalStateFork(DecisionHead):
    name = "fork"
    needs_fork = True

    def __init__(self, d_model: int, hidden: int = 256) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(2 * d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def forward(self, x: HeadInput, return_attribution: bool = False) -> HeadOutput:
        if x.fork is None:
            raise ValueError("FinalStateFork needs fork features (build_head_input(need_fork=True))")
        f = self.norm(x.fork)
        s = _sym_summary(f, x.option_mask).unsqueeze(1).expand_as(f)
        z = self.mlp(torch.cat([f, f - s], -1)).squeeze(-1)
        return HeadOutput(_masked(z, x.option_mask))

    def cost(self, T: int, K: int) -> dict:
        # backbone cost of the fork (K continuations) is priced in train/estimate.py
        return {"flops": 2 * K * self.n_params(), "state_scalars": 0, "attention_free": True,
                "backbone_fork_rows": K}


# ---------------------------------------------------------------- (c) bounded-S snapshot fork
class SnapshotFork(DecisionHead):
    name = "snapshot"
    AGGREGATORS = ("sum", "max", "gated")

    def __init__(self, d_model: int, channels: int = 64, max_snapshots: int = 8, aggregator: str = "gated") -> None:
        super().__init__()
        if aggregator not in self.AGGREGATORS:
            raise ValueError(aggregator)
        self.S, self.agg = max_snapshots, aggregator
        self.inc = IntegratorBank(d_model, channels)
        self.ru = nn.Linear(d_model, channels)
        self.rv = nn.Linear(d_model, channels)
        self.rw = nn.Parameter(torch.randn(channels) / math.sqrt(channels))
        self.gu = nn.Linear(d_model, channels)
        self.gv = nn.Linear(d_model, channels)
        self.gw = nn.Parameter(torch.randn(channels) / math.sqrt(channels))

    def chunk_ids(self, evidence_mask: torch.Tensor) -> torch.Tensor:
        """chunk index in [0, S) per evidence token; -1 elsewhere. Equal-count chunks per row."""
        rank = evidence_mask.long().cumsum(-1) - 1
        n = evidence_mask.sum(-1, keepdim=True).clamp_min(1)
        cid = (rank * self.S) // n
        return torch.where(evidence_mask, cid, torch.full_like(cid, -1))

    def forward(self, x: HeadInput, return_attribution: bool = False) -> HeadOutput:
        bsz, t, _ = x.h.shape
        k = x.e.shape[1]
        qe = x.q_pre.unsqueeze(1).expand(-1, k, -1)
        bias = self.inc.prior(torch.cat([x.e, qe], -1)).squeeze(-1)
        cid = self.chunk_ids(x.evidence_mask)  # [B, T]
        slots = torch.arange(self.S, device=x.device)
        onehot = (cid.unsqueeze(-1) == slots).to(x.h.dtype)  # [B,T,S]; cid = -1 matches no slot
        part = x.h.new_zeros((bsz, self.S, k))
        for t0 in range(0, t, self.inc.t_chunk):
            t1 = min(t, t0 + self.inc.t_chunk)
            d = self.inc.increments(x, t0, t1)  # [B, t, K]
            part = part + torch.einsum("bts,btk->bsk", onehot[:, t0:t1], d)
        # chunk-boundary feature = feature at the last evidence token of each chunk
        pos = torch.arange(t, device=x.device).expand(bsz, t)
        last = torch.full((bsz, self.S), -1, dtype=torch.long, device=x.device)
        for s in range(self.S):
            p = torch.where(cid == s, pos, torch.full_like(pos, -1)).max(-1).values
            last[:, s] = p
        present = last >= 0  # [B, S]
        hb = x.h[torch.arange(bsz, device=x.device).unsqueeze(1), last.clamp_min(0)]  # [B, S, D]
        ve = self.rv(x.e)
        read = F.silu(self.ru(hb).unsqueeze(2) * ve.unsqueeze(1)) @ self.rw / math.sqrt(len(self.rw))  # [B,S,K]
        chunk = (part + read) * present.unsqueeze(-1)
        if self.agg == "sum":
            contrib = chunk
            z = bias + contrib.sum(1)
        elif self.agg == "gated":
            gate = torch.sigmoid(F.silu(self.gu(hb).unsqueeze(2) * self.gv(x.e).unsqueeze(1)) @ self.gw)
            contrib = gate * chunk
            z = bias + contrib.sum(1)
        else:
            masked = chunk.masked_fill(~present.unsqueeze(-1), NEG)
            best = masked.max(1)
            z = bias + best.values
            contrib = torch.zeros_like(chunk).scatter(1, best.indices.unsqueeze(1), best.values.unsqueeze(1))
        return HeadOutput(_masked(z, x.option_mask), bias, contrib if return_attribution else None)

    def cost(self, T: int, K: int) -> dict:
        base = self.inc.cost(T, K)
        return {"flops": base["flops"] + 2 * self.S * K * 4 * len(self.rw), "state_scalars": self.S * K,
                "attention_free": True}


# ---------------------------------------------------------------- (d) Head B
class HeadB(DecisionHead):
    """Option-conditioned selective recurrence over shared backbone features."""

    name = "head_b"

    def __init__(self, d_model: int, d_state: int = 64, hidden: int = 256, t_chunk: int = 128) -> None:
        super().__init__()
        self.d, self.t_chunk = d_state, t_chunk
        self.vproj = nn.Linear(d_model, d_state)
        self.phi = nn.Sequential(nn.Linear(d_model, d_state), nn.SiLU())
        self.cond = nn.Linear(d_model + d_model + d_state, d_state)  # (e_k, q_pre, s)
        self.a_v, self.a_c = nn.Linear(d_state, d_state), nn.Linear(d_state, d_state, bias=False)
        self.b_v, self.b_c = nn.Linear(d_state, d_state), nn.Linear(d_state, d_state, bias=False)
        self.out = nn.Sequential(nn.Linear(3 * d_state, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.constant_(self.a_v.bias, 3.0)  # a ~ 0.95 at init: long memory

    def _setup(self, x: HeadInput):
        s = _sym_summary(self.phi(x.e), x.option_mask)  # [B, d]
        k = x.e.shape[1]
        c = self.cond(torch.cat([x.e, x.q_pre.unsqueeze(1).expand(-1, k, -1), s.unsqueeze(1).expand(-1, k, -1)], -1))
        return s, c  # c: [B, K, d]

    def _gates(self, v: torch.Tensor, c: torch.Tensor, m: torch.Tensor):
        """v [B,t,d], c [B,K,d], m [B,t] -> log_a, b [B,t,K,d]; a=1, b=0 off-evidence."""
        pre_a = self.a_v(v).unsqueeze(2) + self.a_c(c).unsqueeze(1)
        pre_b = self.b_v(v).unsqueeze(2) + self.b_c(c).unsqueeze(1)
        mm = m.unsqueeze(-1).unsqueeze(-1).to(v.dtype)
        log_a = F.logsigmoid(pre_a) * mm
        b = torch.sigmoid(pre_b) * mm
        return log_a, b

    def forward(self, x: HeadInput, return_attribution: bool = False) -> HeadOutput:
        bsz, t, _ = x.h.shape
        s, c = self._setup(x)
        k = c.shape[1]
        r = x.h.new_zeros((bsz, k, self.d))
        for t0 in range(0, t, self.t_chunk):
            t1 = min(t, t0 + self.t_chunk)
            v = self.vproj(x.h[:, t0:t1])
            log_a, b = self._gates(v, c, x.evidence_mask[:, t0:t1])
            cum = log_a.cumsum(1)  # [B,t,K,d]
            total = cum[:, -1]  # log prod over chunk
            # r_end = prod(a) * r_prev + sum_t prod_{u>t} a_u * b_t * v_t   (all factors <= 1: stable)
            decay_after = torch.exp(total.unsqueeze(1) - cum)
            r = torch.exp(total) * r + (decay_after * b * v.unsqueeze(2)).sum(1)
        z = self.out(torch.cat([r, c, s.unsqueeze(1).expand(-1, k, -1)], -1)).squeeze(-1)
        return HeadOutput(_masked(z, x.option_mask))

    @torch.no_grad()
    def reference_scan(self, x: HeadInput) -> torch.Tensor:
        """Token-by-token recurrence (streaming form, O(Kd) state) for testing the chunked scan."""
        s, c = self._setup(x)
        bsz, t, _ = x.h.shape
        r = x.h.new_zeros((bsz, c.shape[1], self.d))
        v_all = self.vproj(x.h)
        for i in range(t):
            log_a, b = self._gates(v_all[:, i : i + 1], c, x.evidence_mask[:, i : i + 1])
            r = torch.exp(log_a[:, 0]) * r + b[:, 0] * v_all[:, i].unsqueeze(1)
        return self.out(torch.cat([r, c, s.unsqueeze(1).expand_as(r)], -1)).squeeze(-1)

    def cost(self, T: int, K: int) -> dict:
        d = self.d
        per_t = 2 * self.vproj.in_features * d + 2 * 2 * d * d  # v_t and its gate terms
        per_tk = 10 * d  # gate adds, sigmoids, multiply-accumulate per option and channel
        return {"flops": T * per_t + T * K * per_tk, "state_scalars": K * d, "attention_free": True}


# ---------------------------------------------------------------- (e) Clef-style control
class _RoutingLayer(nn.Module):
    def __init__(self, width: int, heads: int, ff: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.n1, self.n2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff), nn.GELU(), nn.Linear(ff, width))

    def forward(self, q, mem, mem_pad):
        a, _ = self.attn(self.n1(q), mem, mem, key_padding_mask=mem_pad, need_weights=False)
        q = q + a
        return q + self.ff(self.n2(q))


class ClefCrossAttentionControl(DecisionHead):
    """Clef-shaped control (released joint_head_config sizes: width 1024, 2 routing + 4 layers,
    16 heads, ff 4096). Not Clef's scoring recipe; see the module docstring."""

    name = "clef_control"
    attention_free = False

    def __init__(self, d_model: int, width: int = 1024, routing_layers: int = 2, layers: int = 4,
                 heads: int = 16, feedforward: int = 4096) -> None:
        super().__init__()
        self.width = width
        self.mem = nn.Linear(d_model, width)
        self.qry = nn.Linear(2 * d_model, width)
        self.routing = nn.ModuleList(_RoutingLayer(width, heads, feedforward) for _ in range(routing_layers))
        enc = nn.TransformerEncoderLayer(width, heads, feedforward, dropout=0.0, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))

    def forward(self, x: HeadInput, return_attribution: bool = False) -> HeadOutput:
        k = x.e.shape[1]
        mem = self.mem(x.h)
        q = self.qry(torch.cat([x.e, x.q_end.unsqueeze(1).expand(-1, k, -1)], -1))
        for layer in self.routing:
            q = layer(q, mem, ~x.token_mask)
        q = self.decoder(q, src_key_padding_mask=~x.option_mask)
        return HeadOutput(_masked(self.score(q).squeeze(-1), x.option_mask))

    def cost(self, T: int, K: int) -> dict:
        w, n = self.width, len(self.routing)
        mem = 2 * T * self.mem.in_features * w  # project the sequence once
        kv = n * 2 * 2 * T * w * w  # K and V projections per routing layer
        attn = n * 2 * 2 * K * T * w  # scores + weighted sum
        rest = 2 * K * (self.n_params() - self.mem.weight.numel())
        return {"flops": mem + kv + attn + rest, "state_scalars": n * 2 * T * w,
                "attention_free": False, "note": "keys/values grow with T: softmax over positions"}


HEADS = {
    "integrator": IntegratorBank,
    "fork": FinalStateFork,
    "snapshot": SnapshotFork,
    "head_b": HeadB,
    "clef_control": ClefCrossAttentionControl,
}


def make_head(name: str, d_model: int, **kwargs) -> DecisionHead:
    return HEADS[name](d_model, **kwargs)


@torch.no_grad()
def substitute_token_effect(model_fn, rows, positions, replacement_id: int) -> list:
    """Perturb-and-recompute hook (token SUBSTITUTION, not deletion: the replacement token is
    still read by the backbone): logits with tokens at ``positions`` replaced by
    ``replacement_id``, recomputed through the whole backbone. Compare with booked increments to
    test whether accounting attribution is causal (B-review §3.5). ``model_fn(rows) -> logits``."""
    import copy

    out = []
    for pos in positions:
        mod = []
        for r in rows:
            r2 = copy.copy(r)
            r2.ids = list(r.ids)
            for p in pos:
                r2.ids[p] = replacement_id
            mod.append(r2)
        out.append(model_fn(mod))
    return out
