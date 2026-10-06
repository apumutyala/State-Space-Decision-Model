"""G15 trust-gated reachability (STRUCTURAL): untrusted tokens cannot reach the protected block.

Construction (``TrustGatedBackbone``). Each layer's state is split into two blocks, ``X = (X_P, X_D)``:

* **Protected block P** (the "trusted stream"): the pretrained recurrence run over the TRUSTED tokens only.
  At an untrusted position its transition is the identity and its write is zero (``A_P = I``, ``B_P = 0``),
  and every short-range read (RWKV-7 token shift, GDN causal conv) skips untrusted positions. This is
  implemented exactly by *compaction*: the backbone is run on the trusted subsequence and its per-layer
  outputs are carried forward over untrusted positions (position t reads the output at the last trusted
  position <= t; zero before the first one).
* **Data block D**: the pretrained recurrence over ALL tokens, plus a zero-initialised low-rank bridge per
  layer that adds ``B_l(P_l)`` to the block output. So D may read P; nothing reads D into P.

Joint transition per layer ``[[A_PP, 0], [A_DP, A_DD]]`` (block lower-triangular), untrusted write
``[0; B_D]``. Hence the set reachable from untrusted inputs is contained in ``{0} (+) D`` at every layer,
and, across layers, P features are computed only from P features of the layer below (the P stream never
calls into the D stream). Exactly what is guaranteed (code path + tests, any weights incl. LoRA):

  **G15.** For two inputs with the same TRUSTED token subsequence (any untrusted content, any untrusted
  lengths, any number and placement of untrusted spans), every protected feature at the i-th trusted
  token, at every layer, is bitwise identical, and so is the protected recurrent state. Any readout that
  is a function of protected features only (``readout="protected"``, or the P half of ``"concat"``) is
  therefore invariant to untrusted content.

Scope and caveats (stated, not hidden):
* The guarantee is about the P block. The decision readout in the default ``readout="decision"`` reads D,
  which IS reachable from untrusted inputs (it must be: the evidence is untrusted, reviews/02 section 6).
  G15 does not certify the decision; it guarantees that what is kept in P (schema, question, policy
  framing, earlier trusted records) cannot be overwritten, collapsed (G14) or steered by untrusted text.
* Trust labels come from the host (``decide/provenance``, layout spans); a mislabelled span voids it.
* Determinism: bitwise identity assumes the same device/kernels/batch composition for both inputs (the
  P stream's padded batch shape depends on the longest trusted subsequence in the batch; tests run the
  two inputs in separate batches of one to make the claim exactly about content).
* Channel mixing/MLPs are not masked: they never need to be, because the P stream is a separate
  computation, not a sub-block of one shared residual stream. The price is compute: one extra pass over
  the trusted tokens (``compute_multiplier``). The in-place alternative (a head partition with masked
  output projections, token shift, LoRA gates and FFN) would make the pretrained function unrecoverable
  at init and is not implemented.
* Bridges are zero-initialised, so at init the D stream (and every decision) equals the pretrained model
  bitwise; ``capability_cost`` measures what training the gated arm, or reading P only, costs.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from heads.backbone import HiddenStateBackbone, pad_batch

from . import adapters

READOUTS = ("decision", "concat", "protected")
TRUST_POLICIES = ("state", "caller")
GUARANTEE_ID = "ssdm.g15.trust_gated_reachability.v1"


class Bridge(nn.Module):
    """Low-rank P -> D bridge; params named ``lora_A`` / ``lora_B`` so the trainer's LoRA group trains them."""

    def __init__(self, d: int, rank: int, generator: torch.Generator | None = None) -> None:
        super().__init__()
        a = torch.empty(rank, d, dtype=torch.float32)
        nn.init.kaiming_uniform_(a, a=5 ** 0.5, generator=generator)
        self.lora_A = nn.Parameter(a)
        self.lora_B = nn.Parameter(torch.zeros(d, rank, dtype=torch.float32))

    def forward(self, p: torch.Tensor) -> torch.Tensor:
        return (p @ self.lora_A.to(p.dtype).t()) @ self.lora_B.to(p.dtype).t()


def compaction(mask: torch.Tensor, trust: torch.Tensor) -> tuple[list[list[int]], torch.Tensor]:
    """Per row: trusted positions, and ``cnt[b, t]`` = index (in trusted order) of the last trusted position
    <= t, or -1 if none."""
    keep = mask.bool() & trust.bool()
    cnt = keep.long().cumsum(-1) - 1
    pos = [torch.nonzero(row, as_tuple=False).flatten().tolist() for row in keep]
    return pos, cnt


def carry(pc: torch.Tensor, cnt: torch.Tensor) -> torch.Tensor:
    """Scatter compacted ``[B, Tc, d]`` features to ``[B, T, d]`` with carry-forward (identity transition)."""
    idx = cnt.clamp_min(0).unsqueeze(-1).expand(-1, -1, pc.shape[-1])
    if pc.shape[1] == 0:
        return pc.new_zeros(cnt.shape + (pc.shape[-1],))
    out = torch.gather(pc, 1, idx)
    return torch.where((cnt >= 0).unsqueeze(-1), out, torch.zeros_like(out))


def row_untrusted_spans(row, policy: str = "state") -> list[tuple[int, int]]:
    """Untrusted token spans of a ``heads.layout.LayoutRow`` (fail closed if the row cannot say)."""
    meta = getattr(row, "meta", {}) or {}
    key = "untrusted_spans" if policy == "state" else "caller_spans"
    if key not in meta:
        if policy == "state" and meta.get("evidence") == "state":
            return [tuple(row.state_span)]
        raise ValueError(f"row has no meta[{key!r}]: rebuild it with heads.layout.build_row (trust gate fails closed)")
    return [tuple(s) for s in meta[key]]


def row_trust(row, policy: str = "state", length: int | None = None) -> list[bool]:
    n = len(row.ids) if length is None else length
    t = [True] * n
    for a, b in row_untrusted_spans(row, policy):
        for i in range(a, min(b, n)):
            t[i] = False
    return t


class TrustGatedBackbone(HiddenStateBackbone):
    """``features(ids, mask, trust)``: see module docstring. ``trust`` is ``[B, T]`` bool (True = trusted);
    ``None`` means all trusted (P == D at init)."""

    wants_trust = True

    def __init__(self, base: HiddenStateBackbone, bridge_rank: int = 8, readout: str = "decision",
                 trust_policy: str = "state", seed: int = 0) -> None:
        if base.capture != "hidden_states":
            raise ValueError("trust gate needs capture='hidden_states' (RWKV-7 HF / Qwen3.5 GDN)")
        if readout not in READOUTS or trust_policy not in TRUST_POLICIES:
            raise ValueError((readout, trust_policy))
        super().__init__(base.model, capture="hidden_states", layer=base.layer, spec=base.spec,
                         pad_id=base.pad_id, fork_mode=base.fork_mode)
        adapters.family_of(base.model)
        self.readout, self.trust_policy, self.bridge_rank = readout, trust_policy, bridge_rank
        self.__dict__["_blocks"] = list(adapters.blocks(base.model))  # not re-registered as submodules
        d = base.d_model
        gen = torch.Generator().manual_seed(seed * 7919 + 11)
        self.bridges = nn.ModuleList([Bridge(d, bridge_rank, gen) for _ in self._blocks])
        self.bridges.to(next(base.model.parameters()).device)
        self._mode: str | None = None
        self._rec: list[torch.Tensor | None] = [None] * len(self._blocks)
        self._inject: list[torch.Tensor | None] = [None] * len(self._blocks)
        self._handles = [b.register_forward_hook(self._make_hook(i)) for i, b in enumerate(self._blocks)]
        self.last_protected: list[torch.Tensor] | None = None  # per-layer carried P features [B, T, d]
        self.last_protected_final: torch.Tensor | None = None
        self.tokens_processed = 0
        self.provenance = getattr(base, "provenance", None)  # lineage: Q9-AF provenance passes through
        self.controlsafe = self.identity()

    def identity(self) -> dict[str, Any]:
        return {"guarantee": GUARANTEE_ID, "readout": self.readout, "trust_policy": self.trust_policy,
                "bridge_rank": self.bridge_rank, "family": adapters.family_of(self.model)}

    @property
    def d_model(self) -> int:
        d = super().d_model
        return 2 * d if getattr(self, "readout", "decision") == "concat" else d

    def trainable_bridge_parameters(self) -> list[nn.Parameter]:
        return list(self.bridges.parameters())

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, i: int):
        def fn(module, args, output):
            if self._mode is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if self._mode == "record":
                self._rec[i] = h
                return None
            p = self._inject[i]
            if p is None:
                return None
            h2 = h + self.bridges[i](p.to(h.dtype))
            return (h2,) + tuple(output[1:]) if isinstance(output, tuple) else h2
        return fn

    def _run(self, ids: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        kw: dict[str, Any] = {"input_ids": ids, "output_hidden_states": True, "use_cache": False}
        if mask is not None:
            kw["attention_mask"] = mask
        return self.model(**kw).hidden_states[self.layer]

    # ------------------------------------------------------------------ streams
    def protected_stream(self, ids: torch.Tensor, mask: torch.Tensor, trust: torch.Tensor):
        """P stream. The only token ids that enter it are ``ids[b, pos[b]]`` (trusted positions): the
        untrusted ids are never read. Returns (per-layer carried features, carried final features)."""
        pos, cnt = compaction(mask, trust)
        dev = ids.device
        n_layers = len(self._blocks)
        trusted_ids = [[int(ids[b, j]) for j in p] for b, p in enumerate(pos)]
        if max((len(s) for s in trusted_ids), default=0) == 0:
            zero = torch.zeros(ids.shape + (super().d_model,), device=dev)
            return [zero] * n_layers, zero
        nonempty = [s if s else [self.pad_id] for s in trusted_ids]
        cids, cmask = pad_batch(nonempty, self.pad_id, device=dev)
        for b, s in enumerate(trusted_ids):
            if not s:
                cmask[b, 0] = False
        self.tokens_processed += int(cmask.sum())
        self._mode = "record"
        try:
            final = self._run(cids, cmask)
        finally:
            self._mode = None
        layers = [carry(self._rec[i], cnt) for i in range(n_layers)]
        self._rec = [None] * n_layers
        return layers, carry(final, cnt)

    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 trust: torch.Tensor | None = None) -> torch.Tensor:
        dev = self.device
        ids = input_ids.to(dev)
        mask = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.to(dev).bool()
        trust = torch.ones_like(mask) if trust is None else trust.to(dev).bool()
        p_layers, p_final = self.protected_stream(ids, mask, trust)
        self.last_protected, self.last_protected_final = p_layers, p_final
        if self.readout == "protected":
            return p_final
        self._inject = p_layers
        self._mode = "inject"
        self.tokens_processed += int(mask.sum())
        try:
            d = self._run(ids, mask)
        finally:
            self._mode = None
            self._inject = [None] * len(self._blocks)
        if self.readout == "concat":
            return torch.cat([d, p_final.to(d.dtype)], dim=-1)
        return d

    def forward(self, input_ids, attention_mask=None, trust=None):  # pragma: no cover - alias
        return self.features(input_ids, attention_mask, trust)

    def fork_features(self, prefix: Sequence[int], conts: Sequence[Sequence[int]], trust: Sequence[bool] | None = None):
        """Fork by recompute (continuations are trusted template tokens). ``fork_mode`` is ignored: the
        two-stream cache fork is not implemented, and recompute is exact."""
        tr = list(trust) if trust is not None else [True] * len(prefix)
        seqs = [list(prefix)] + [list(prefix) + list(c) for c in conts]
        trs = [tr] + [tr + [True] * len(c) for c in conts]
        ids, mask = pad_batch(seqs, self.pad_id, device=self.device)
        tmask, _ = pad_batch([[int(x) for x in t] for t in trs], 0, device=self.device)
        h = self.features(ids, mask, tmask.bool())
        last = mask.sum(-1) - 1
        return h[0, : len(prefix)], h[torch.arange(1, len(seqs), device=self.device), last[1:]]


def wrap(backbone: HiddenStateBackbone, cfg: dict | None) -> HiddenStateBackbone:
    """Config-flag entry point (train/ and safety_eval): ``{"trust_gate": true, "bridge_rank": 8,
    "readout": "decision", "trust_policy": "state"}``; anything falsy returns the backbone unchanged."""
    if not cfg or not cfg.get("trust_gate"):
        return backbone
    return TrustGatedBackbone(backbone, bridge_rank=int(cfg.get("bridge_rank", 8)),
                              readout=cfg.get("readout", "decision"), trust_policy=cfg.get("trust_policy", "state"),
                              seed=int(cfg.get("seed", 0)))


# ----------------------------------------------------------------------------------- checks and costs
@torch.no_grad()
def reachability_check(gated: TrustGatedBackbone, ids: Sequence[int], trust: Sequence[bool], n_trials: int = 8,
                       vocab: tuple[int, int] = (4, 260), seed: int = 0, vary_length: bool = True) -> dict:
    """Perturb (and resize) the untrusted spans; compare protected features in trusted order, bitwise.
    Returns counts; ``protected_bitwise_equal`` must be True for every trial (STRUCTURAL guarantee), and
    ``decision_changed`` > 0 shows the test is not vacuous (the D stream does see untrusted text)."""
    g = torch.Generator().manual_seed(seed)
    ids, trust = list(ids), list(trust)

    def run(x, t):
        xi = torch.tensor([x]); ti = torch.tensor([t])
        d = gated.features(xi, None, ti)
        keep = [i for i, z in enumerate(t) if z]
        prot = [pl[0, keep].clone() for pl in gated.last_protected] + [gated.last_protected_final[0, keep].clone()]
        return d[0, -1].clone(), prot

    d0, p0 = run(ids, trust)
    ok, changed = 0, 0
    for _ in range(n_trials):
        x, t, i = [], [], 0
        while i < len(ids):
            if trust[i]:
                x.append(ids[i]); t.append(True); i += 1
                continue
            j = i
            while j < len(ids) and not trust[j]:
                j += 1
            n = j - i
            if vary_length:
                n = max(1, n + int(torch.randint(-n // 2, n + 1, (1,), generator=g)))
            x += torch.randint(vocab[0], vocab[1], (n,), generator=g).tolist(); t += [False] * n
            i = j
        d1, p1 = run(x, t)
        ok += all(torch.equal(a, b) for a, b in zip(p0, p1))
        changed += not torch.equal(d0, d1)
    return {"guarantee": GUARANTEE_ID, "trials": n_trials, "protected_bitwise_equal": ok,
            "decision_feature_changed": changed, "pass": ok == n_trials}


def capability_cost(base_fn, gated_fn, items: Sequence, metric) -> dict:
    """Paired capability cost of the gated arm. ``base_fn(item)`` / ``gated_fn(item)`` return the per-item
    quantity ``metric(pred, item) -> float`` scores (e.g. NLL or 0/1 correct). Returns paired mean delta
    and per-item deltas (gated - base); bootstrap / noninferiority is ``harness.bootstrap.compare``'s job."""
    deltas = [metric(gated_fn(it), it) - metric(base_fn(it), it) for it in items]
    n = len(deltas)
    return {"n": n, "mean_delta": sum(deltas) / n if n else 0.0, "deltas": deltas}


def compute_multiplier(gated: TrustGatedBackbone, n_tokens: int) -> float:
    """Tokens processed by both streams per token of input since the counter was reset."""
    return gated.tokens_processed / max(1, n_tokens)
