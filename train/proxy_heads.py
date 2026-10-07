"""Proxy decision heads over CACHED frozen-Q9 features (jobs/q9_feature_cache; rule: train/proxy_rule.py).

Input: one REQUEST (all its cached questions) as a ``ProxyRequest``. Output: per question, typed log-probabilities
over its options (``heads.typed.TypedOutput``, the m2 typed layer: noul / choice softmax, score cumulative link).

Families
  (a) ``fork``      the m2 fork head (``heads.heads.FinalStateFork``, hidden 256) on the cached fork features: the exact
                    module m2 cells train, so a proxy fork number is the m2 head-only number (rank 0, frozen backbone).
  (b) ``mixer``     ATTENTION-FREE cross-field / option mixer, the attention-free analogue of Clef's two-stage routing
                    head. Per option: projections of fork / option-span / lexical (output-embedding) features + the
                    question vectors + LM answer log-prob features (E4) + key-class and qtype embeddings. Then
                      * option-conditioned EVIDENCE ROUTING by a gated linear recurrence over the cached evidence bank
                        (<= 32 chunk summaries of the state, in order): r_k <- a_{s,k} * r_k + b_{s,k} * v_s with
                        independent sigmoid gates (Head-B style; no normalisation across chunks: no attention);
                      * option mixing inside a question (DeepSets: each option sees the mean of its siblings);
                      * FIELD mixing across the questions of the request (mean + max pooled field summaries fed back,
                        two rounds; order-invariant; a fixed set of Q vectors, not positions);
                      * score = MLP(h_k) + lexical prior (scaled cosine of option lexical vs question, Clef's prior form)
                        + a learned, feature-gated pool weight times the LM option log-prob (E4).
                    Flags ``no_lm`` / ``no_keyclass`` / ``no_route`` / ``no_fields`` / ``no_lex`` ablate one factor each.
  (c) ``attn_ref``  ATTENTION reference (NOT attention-free; never a candidate; ``attention_free = False``): the same
                    option inputs, 2 cross-attention routing layers over the evidence bank, then 2 self-attention
                    layers over every option of every question of the request (Clef's joint field step), lexical
                    prior. Width 256, so it is a smaller-than-Clef upper reference (Clef's head: ~128M params).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from heads.option_keys import KEY_CLASSES
from heads.typed import TypedOutput

QTYPE_IDS = {"noul": 0, "choice": 1, "score": 2}
KEY_IDS = {k: i for i, k in enumerate(KEY_CLASSES)}


@dataclass
class ProxyQuestion:
    qid: str
    qtype: str
    option_ids: list[str]
    fork: torch.Tensor  # [K, D]
    e: torch.Tensor  # [K, D]
    lex: torch.Tensor  # [K, D]
    q_pre: torch.Tensor  # [D]
    q_end: torch.Tensor  # [D]
    lm_lp: torch.Tensor  # [K] fp32
    lm_ntok: torch.Tensor  # [K]
    key_class: str
    bank_key: str | None = None
    label: int = -1

    @property
    def k(self) -> int:
        return len(self.option_ids)


@dataclass
class ProxyRequest:
    rid: str
    split: str
    questions: list[ProxyQuestion]
    banks: dict[str, torch.Tensor] = field(default_factory=dict)
    catalog_id: int | None = None
    skipped: dict = field(default_factory=dict)
    #: longest cached row's model input (tokens; the record's per-question ``T``), None for records without it
    max_tokens: int | None = None


def from_record(rec: dict, lex_table: dict | None = None) -> ProxyRequest:
    """A cached request record (jobs.q9_feature_cache.extract) -> ProxyRequest (tensors stay bf16 on CPU). The lexical
    feature is rebuilt from the arm's ``lex_vocab.pt`` table and the stored option token ids (the cache does not store
    it per option; jobs/q9_feature_cache review B1)."""
    from jobs.q9_feature_cache.extract import lexical

    def lex(q):
        if "lex" in q:  # legacy records
            return q["lex"]
        if lex_table is None:
            raise ValueError("records without a stored lexical feature need the arm's lex_vocab.pt table")
        return lexical(q, lex_table)

    qs = [ProxyQuestion(q["qid"], q["qtype"], list(q["option_ids"]), q["fork"], q["e"], lex(q), q["q_pre"],
                        q["q_end"], q["lm_lp"].float(), q["lm_ntok"].float(), q["key_class"], q.get("bank_key"),
                        int(q.get("label", -1))) for q in rec["questions"]]
    ts = [int(q["T"]) for q in rec["questions"] if "T" in q]
    return ProxyRequest(rec["rid"], rec["split"], qs, dict(rec.get("banks") or {}), rec.get("catalog_id"),
                        dict(rec.get("skipped") or {}), max(ts) if ts else None)


def _f(x: torch.Tensor) -> torch.Tensor:
    return x.float()


def lm_features(q: ProxyQuestion) -> torch.Tensor:
    """[K, 4]: summed log-lik, per-token log-lik, log-softmax over options of the summed log-lik (the stock readout's
    own log-probability), and its centred version."""
    lp = q.lm_lp
    ls = torch.log_softmax(lp, -1)
    return torch.stack([lp / 10.0, lp / q.lm_ntok.clamp_min(1), ls, ls - ls.mean()], -1)


class ProxyHead(nn.Module):
    family = "base"
    attention_free = True

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d = d_model
        self.typed = TypedOutput(d_model)

    def logits(self, req: ProxyRequest) -> list[torch.Tensor]:
        raise NotImplementedError

    def forward(self, req: ProxyRequest) -> list[torch.Tensor]:
        """Per question: typed log-probabilities [K]."""
        out = []
        for q, z in zip(req.questions, self.logits(req)):
            lp = self.typed(z.unsqueeze(0), [q.qtype], [q.k], _f(q.e).unsqueeze(0))[0]
            out.append(lp)
        return out

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class ForkProxy(ProxyHead):
    """(a) the m2 FinalStateFork head, unchanged, on cached fork features."""

    family = "fork"

    def __init__(self, d_model: int, hidden: int = 256, use_lm: bool = False) -> None:
        super().__init__(d_model)
        from heads.heads import FinalStateFork

        self.head = FinalStateFork(d_model, hidden)
        self.use_lm = use_lm
        if use_lm:
            self.lm_w = nn.Parameter(torch.zeros(()))

    def logits(self, req: ProxyRequest) -> list[torch.Tensor]:
        from heads.features import HeadInput

        out = []
        for q in req.questions:
            f = _f(q.fork).unsqueeze(0)
            x = HeadInput(h=f.new_zeros((1, 1, self.d)), token_mask=torch.ones((1, 1), dtype=torch.bool),
                          evidence_mask=torch.ones((1, 1), dtype=torch.bool), q_pre=_f(q.q_pre)[None],
                          q_end=_f(q.q_end)[None], e=_f(q.e)[None], option_mask=torch.ones((1, q.k), dtype=torch.bool),
                          fork=f, ks=[q.k])
            z = self.head(x).logits[0]
            if self.use_lm:
                z = z + self.lm_w * torch.log_softmax(q.lm_lp, -1)
            out.append(z)
        return out


class _OptionEncoder(nn.Module):
    """Shared per-option input embedding of the mixer and the attention reference."""

    def __init__(self, d_model: int, width: int, use_lm: bool = True, use_keyclass: bool = True) -> None:
        super().__init__()
        self.use_lm, self.use_keyclass = use_lm, use_keyclass
        self.n_fork, self.n_e, self.n_lex = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.n_q = nn.LayerNorm(d_model)
        self.p_fork, self.p_e = nn.Linear(d_model, width), nn.Linear(d_model, width, bias=False)
        self.p_lex = nn.Linear(d_model, width, bias=False)
        self.p_qend, self.p_qpre = nn.Linear(d_model, width, bias=False), nn.Linear(d_model, width, bias=False)
        self.p_lm = nn.Linear(4, width, bias=False)
        self.qtype = nn.Embedding(len(QTYPE_IDS), width)
        self.keycls = nn.Embedding(len(KEY_IDS), width)
        self.out = nn.Sequential(nn.GELU(), nn.Linear(width, width))
        self.norm = nn.LayerNorm(width)

    def question_vec(self, q: ProxyQuestion) -> torch.Tensor:
        v = self.p_qend(self.n_q(_f(q.q_end))) + self.p_qpre(self.n_q(_f(q.q_pre))) + self.qtype.weight[QTYPE_IDS[q.qtype]]
        if self.use_keyclass:
            v = v + self.keycls.weight[KEY_IDS[q.key_class]]
        return v

    def forward(self, q: ProxyQuestion) -> tuple[torch.Tensor, torch.Tensor]:
        qv = self.question_vec(q)
        h = self.p_fork(self.n_fork(_f(q.fork))) + self.p_e(self.n_e(_f(q.e))) + self.p_lex(self.n_lex(_f(q.lex)))
        if self.use_lm:
            h = h + self.p_lm(lm_features(q))
        h = h + qv
        return self.norm(h + self.out(h)), qv


def _bank(req: ProxyRequest, q: ProxyQuestion, d: int) -> torch.Tensor | None:
    if q.bank_key is None or q.bank_key not in req.banks:
        return None
    return _f(req.banks[q.bank_key])


class AFMixer(ProxyHead):
    """(b) attention-free routing + option + field mixer (see module docstring)."""

    family = "mixer"

    def __init__(self, d_model: int, width: int = 256, field_rounds: int = 2, no_lm: bool = False,
                 no_keyclass: bool = False, no_route: bool = False, no_fields: bool = False, no_lex: bool = False) -> None:
        super().__init__(d_model)
        self.flags = {"no_lm": no_lm, "no_keyclass": no_keyclass, "no_route": no_route, "no_fields": no_fields,
                      "no_lex": no_lex}
        self.enc = _OptionEncoder(d_model, width, use_lm=not no_lm, use_keyclass=not no_keyclass)
        # evidence routing: gated linear recurrence over bank chunks, conditioned on the option
        self.n_bank = nn.LayerNorm(d_model)
        self.v = nn.Linear(d_model, width)
        self.a_v, self.a_h = nn.Linear(width, width), nn.Linear(width, width, bias=False)
        self.b_v, self.b_h = nn.Linear(width, width), nn.Linear(width, width, bias=False)
        nn.init.constant_(self.a_v.bias, 2.0)  # long memory at init
        self.route_out = nn.Linear(width, width)
        # option mixing (DeepSets) and field mixing
        self.opt_mix = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, width))
        self.field_mix = nn.ModuleList(nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, width))
                                       for _ in range(field_rounds))
        self.n1 = nn.LayerNorm(width)
        self.score = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))
        # lexical prior (Clef form: scale * cos(lexical option, question)) and the E4 gated pool
        self.lex_o, self.lex_q = nn.Linear(d_model, width, bias=False), nn.Linear(width, width, bias=False)
        self.lex_scale = nn.Parameter(torch.tensor(1.0))
        self.lm_gate = nn.Linear(width, 1)
        nn.init.constant_(self.lm_gate.bias, 0.0)

    def _route(self, h: torch.Tensor, bank: torch.Tensor | None) -> torch.Tensor:
        if bank is None or self.flags["no_route"]:
            return h
        v = self.v(self.n_bank(bank))  # [S, W]
        ah, bh = self.a_h(h), self.b_h(h)  # [K, W]
        r = torch.zeros_like(h)
        for s in range(v.shape[0]):  # in evidence order; O(S K W), no normalisation over chunks
            a = torch.sigmoid(self.a_v(v[s]) + ah)
            b = torch.sigmoid(self.b_v(v[s]) + bh)
            r = a * r + b * v[s]
        return h + self.route_out(r)

    def logits(self, req: ProxyRequest) -> list[torch.Tensor]:
        hs, qvs = [], []
        for q in req.questions:
            h, qv = self.enc(q)
            h = self._route(h, _bank(req, q, self.d))
            h = h + self.opt_mix(torch.cat([h, h - h.mean(0, keepdim=True)], -1))
            hs.append(h)
            qvs.append(qv)
        if not self.flags["no_fields"] and len(hs) > 1:
            for layer in self.field_mix:
                f = torch.stack([h.mean(0) for h in hs])  # [Q, W] field summaries
                g_mean, g_max = f.mean(0), f.max(0).values
                hs = [h + layer(torch.cat([h, g_mean.expand_as(h), g_max.expand_as(h)], -1)) for h in hs]
        out = []
        for q, h, qv in zip(req.questions, hs, qvs):
            z = self.score(self.n1(h)).squeeze(-1)
            if not self.flags["no_lex"]:
                lo = F.normalize(self.lex_o(self.enc.n_lex(_f(q.lex))), dim=-1)
                lq = F.normalize(self.lex_q(qv), dim=-1)
                z = z + self.lex_scale * (lo @ lq)
            if not self.flags["no_lm"]:
                z = z + F.softplus(self.lm_gate(qv)) * torch.log_softmax(q.lm_lp, -1)
            out.append(z)
        return out


class AttnRef(ProxyHead):
    """(c) ATTENTION reference: cross-attention over the bank, then self-attention over all options of the request.
    NOT attention-free; an upper-bound reference only (train/proxy_rule.py: never a candidate)."""

    family = "attn_ref"
    attention_free = False

    def __init__(self, d_model: int, width: int = 256, heads: int = 4, routing_layers: int = 2, joint_layers: int = 2) -> None:
        super().__init__(d_model)
        self.enc = _OptionEncoder(d_model, width)
        self.n_bank = nn.LayerNorm(d_model)
        self.mem = nn.Linear(d_model, width)
        self.route = nn.ModuleList(nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(routing_layers))
        self.rn = nn.ModuleList(nn.LayerNorm(width) for _ in range(routing_layers))
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.joint = nn.TransformerEncoder(layer, joint_layers, enable_nested_tensor=False)
        self.score = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 1))
        self.lex_o, self.lex_q = nn.Linear(d_model, width, bias=False), nn.Linear(width, width, bias=False)
        self.lex_scale = nn.Parameter(torch.tensor(1.0))

    def logits(self, req: ProxyRequest) -> list[torch.Tensor]:
        hs, qvs = [], []
        for q in req.questions:
            h, qv = self.enc(q)
            bank = _bank(req, q, self.d)
            if bank is not None:
                m = self.mem(self.n_bank(bank)).unsqueeze(0)
                x = h.unsqueeze(0)
                for att, n in zip(self.route, self.rn):
                    a, _ = att(n(x), m, m, need_weights=False)
                    x = x + a
                h = x[0]
            hs.append(h)
            qvs.append(qv)
        sizes = [h.shape[0] for h in hs]
        j = self.joint(torch.cat(hs, 0).unsqueeze(0))[0]
        out = []
        for q, h, qv in zip(req.questions, j.split(sizes), qvs):
            lo = F.normalize(self.lex_o(self.enc.n_lex(_f(q.lex))), dim=-1)
            z = self.score(h).squeeze(-1) + self.lex_scale * (lo @ F.normalize(self.lex_q(qv), dim=-1))
            out.append(z)
        return out


FAMILIES = {"fork": ForkProxy, "mixer": AFMixer, "attn_ref": AttnRef}
#: recipe keys (train/proxy_rule.RECIPES) that are head kwargs
HEAD_FLAGS = ("no_lm", "no_keyclass", "no_route", "no_fields", "no_lex")


def make_proxy_head(recipe: dict, d_model: int, seed: int) -> ProxyHead:
    """Parameters initialised from ``seed`` in a forked RNG (as heads.model.DecisionModel)."""
    kw = {k: True for k in HEAD_FLAGS if recipe.get(k)}
    kw.update(recipe.get("head_kwargs") or {})
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return FAMILIES[recipe["family"]](d_model, **kw)


def describe(head: ProxyHead) -> dict:
    return {"family": head.family, "attention_free": head.attention_free, "params": head.n_params(),
            "flags": getattr(head, "flags", {}), "width_note": "d_model=%d" % head.d, "log2_params": round(math.log2(max(head.n_params(), 1)), 2)}
