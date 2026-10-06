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
* Trust labels come from the host; a mislabelled span voids it. Policies (``row_trust``):
  ``provenance`` (default): a caller segment is trusted only if the host declared its request field
  instruction-trusted in ``request["ssdm"]["provenance"]`` (complete tiling, all ``instruction``); ids
  and undeclared fields are data, so an untrusted passage placed in question instructions (m2
  ``placement: instructions``) is untrusted wherever it sits. ``state``: only the rendered state is
  untrusted -- sound ONLY if the host puts all untrusted text in ``state``; it fails closed on rows whose
  evidence moved to the instructions (``meta["evidence"] == "reread_instructions"``). ``caller``: every
  non-template token is untrusted (P then holds template text only). Continuations and stand-alone
  option sequences are labelled by the same policy (option ids are data under ``provenance``/``caller``).
  ``features`` refuses ``trust=None`` (fail closed).
* Determinism: the protected features are independent of untrusted content and lengths ALWAYS (also of a
  batch-mate's untrusted content/length); they are bitwise identical for identical host-controlled batch
  shape (number of DISTINCT trusted subsequences in the batch -- identical ones run once -- and the longest
  one): the kernels' chunking and GDN's batched reductions
  can move the last bits when that shape changes.
* Training scope: G15 is an INFERENCE-TIME reachability statement. LoRA is shared by both streams and is
  trained on losses that read untrusted training text, so P's function (its weights) does depend on
  untrusted TRAINING data; poisoning of training data is G6/G12 territory, not G15.
* Gradient checkpointing is refused (the bridge injection lives in forward hooks that HF recompute would
  not replay consistently).
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
TRUST_POLICIES = ("provenance", "state", "caller")
STREAM_FORKS = ("recompute", "cache")
DEFAULT_POLICY = "provenance"
ID_PATH = "#ids"  # == heads.layout.ID_PATH
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


class TrustUnavailable(ValueError):
    """The row cannot say which tokens are untrusted under the requested policy: fail closed."""


def _meta(row, key):
    meta = getattr(row, "meta", {}) or {}
    if key not in meta:
        raise TrustUnavailable(f"row has no meta[{key!r}]: rebuild it with heads.layout.build_row (trust gate "
                               "fails closed)")
    return meta[key]


def _untrusted_by_paths(spans, policy: str, trusted_paths) -> list[tuple[int, int]]:
    if policy == "caller":
        return [(a, b) for a, b, _ in spans]
    ok = set(trusted_paths)
    return [(a, b) for a, b, path in spans if path == ID_PATH or path not in ok]


def row_untrusted_spans(row, policy: str = DEFAULT_POLICY) -> list[tuple[int, int]]:
    """Untrusted token spans of a ``heads.layout.LayoutRow`` (fail closed if the row cannot say)."""
    if policy not in TRUST_POLICIES:
        raise ValueError(policy)
    meta = getattr(row, "meta", {}) or {}
    if policy == "state":
        if meta.get("evidence") != "state":
            # m2 ``placement: instructions``: the passage is in the question instructions, which the "state"
            # policy would label trusted. Refuse instead of silently protecting injected text.
            raise TrustUnavailable("trust_policy='state' is unsound for this row (evidence is "
                                   f"{meta.get('evidence')!r}, not the state): use trust_policy='provenance'")
        return [tuple(s) for s in _meta(row, "untrusted_spans")]
    return _untrusted_by_paths(_meta(row, "field_spans"), policy, meta.get("trusted_paths", []))


def _mask(n: int, spans) -> list[bool]:
    t = [True] * n
    for a, b in spans:
        for i in range(a, min(b, n)):
            t[i] = False
    return t


def row_trust(row, policy: str = DEFAULT_POLICY, length: int | None = None) -> list[bool]:
    return _mask(len(row.ids) if length is None else length, row_untrusted_spans(row, policy))


def continuation_trust(row, k: int, policy: str = DEFAULT_POLICY) -> list[bool]:
    """Trust of option continuation k (``" <option id>"`` + template terminator). Under ``state`` the
    continuation holds no state text (trusted); under ``provenance``/``caller`` the option id is data."""
    n = len(row.continuations[k])
    if policy == "state":
        row_untrusted_spans(row, policy)  # the same fail-closed check as the prefix
        return [True] * n
    return _mask(n, _untrusted_by_paths(_meta(row, "cont_spans")[k], policy, row.meta.get("trusted_paths", [])))


def option_seq_trust(row, k: int, policy: str = DEFAULT_POLICY) -> list[bool]:
    """Trust of the symmetric layout's stand-alone option sequence k."""
    n = len(row.option_seqs[k])
    if policy == "state":
        row_untrusted_spans(row, policy)
        return [True] * n
    spans = _meta(row, "option_seq_spans")[k]
    return _mask(n, _untrusted_by_paths(spans, policy, row.meta.get("trusted_paths", [])))


class TrustGatedBackbone(HiddenStateBackbone):
    """``features(ids, mask, trust)``: see module docstring. ``trust`` is ``[B, T]`` bool (True = trusted)
    and is REQUIRED (``None`` raises: an unlabelled call must not silently treat everything as trusted)."""

    wants_trust = True

    def __init__(self, base: HiddenStateBackbone, bridge_rank: int = 8, readout: str = "decision",
                 trust_policy: str = DEFAULT_POLICY, seed: int = 0, stream_fork: str = "recompute") -> None:
        if stream_fork not in STREAM_FORKS:
            raise ValueError(f"stream_fork must be one of {STREAM_FORKS}, got {stream_fork!r}")
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
        self.p_cache_hits = 0
        self.p_rows_deduplicated = 0
        self.stream_fork = stream_fork
        self._inject_seq: list[list[torch.Tensor]] | None = None
        self._calls: list[int] = [0] * len(self._blocks)
        self.last_protected_compact: dict | None = None
        self.provenance = getattr(base, "provenance", None)  # lineage: Q9-AF provenance passes through
        self.controlsafe = self.identity()

    def identity(self) -> dict[str, Any]:
        ident = {"guarantee": GUARANTEE_ID, "readout": self.readout, "trust_policy": self.trust_policy,
                 "bridge_rank": self.bridge_rank, "family": adapters.family_of(self.model)}
        if getattr(self, "stream_fork", "recompute") != "recompute":  # old identities unchanged
            ident["stream_fork"] = self.stream_fork
        return ident

    @property
    def d_model(self) -> int:
        d = super().d_model
        return 2 * d if getattr(self, "readout", "decision") == "concat" else d

    def trainable_bridge_parameters(self) -> list[nn.Parameter]:
        return list(self.bridges.parameters())

    def enable_gradient_checkpointing(self) -> bool:
        raise RuntimeError("controlsafe trust gate: gradient checkpointing is not supported (the P->D bridge "
                           "injection lives in forward hooks); train the gated arm without checkpointing")

    def _check_no_checkpointing(self) -> None:
        if torch.is_grad_enabled() and (getattr(self.model, "is_gradient_checkpointing", False)
                                        or any(getattr(m, "gradient_checkpointing", False)
                                               for m in self.model.modules())):
            raise RuntimeError("controlsafe trust gate: the wrapped model has gradient checkpointing enabled; "
                               "disable it (gradient_checkpointing_disable) for the gated arm")

    # ------------------------------------------------------------------ hooks
    def _make_hook(self, i: int):
        def fn(module, args, output):
            if self._mode is None:
                return None
            h = output[0] if isinstance(output, tuple) else output
            if self._mode == "record":
                self._rec[i] = h
                return None
            if self._mode == "inject_seq":  # cache fork: call c of this layer reads injection c (prefix, continuations)
                c = self._calls[i]
                self._calls[i] += 1
                if self._inject_seq is None or c >= len(self._inject_seq):
                    raise RuntimeError("trust-gate cache fork: unexpected extra model call (fail closed)")
                p = self._inject_seq[c][i]
                if p.shape[:2] != h.shape[:2]:
                    raise RuntimeError(f"trust-gate cache fork: injection {tuple(p.shape[:2])} != block output "
                                       f"{tuple(h.shape[:2])} (fail closed)")
            else:
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
    def protected_stream(self, ids: torch.Tensor, mask: torch.Tensor, trust: torch.Tensor, p_cache: dict | None = None):
        """P stream. The only token ids that enter it are ``ids[b, pos[b]]`` (trusted positions): the
        untrusted ids are never read. Returns (per-layer carried features, carried final features).

        ``p_cache`` (a previous call's ``last_protected_compact``) is reused, without running P, iff its trusted
        id lists equal this call's exactly: P is a function of those lists only, so reuse is exact (G4 over E)."""
        pos, cnt = compaction(mask, trust)
        dev = ids.device
        n_layers = len(self._blocks)
        trusted_ids = [[int(ids[b, j]) for j in p] for b, p in enumerate(pos)]
        if p_cache is not None and p_cache["trusted_ids"] == trusted_ids:
            self.p_cache_hits += 1
            self.last_protected_compact = p_cache
            return [carry(x, cnt) for x in p_cache["layers"]], carry(p_cache["final"], cnt)
        if max((len(s) for s in trusted_ids), default=0) == 0:
            zero = torch.zeros(ids.shape + (super().d_model,), device=dev)
            return [zero] * n_layers, zero
        # Rows with IDENTICAL trusted id lists (e.g. the K option forks of one row under ``provenance``: the option
        # ids are data, so every fork has the same trusted subsequence) run once and are broadcast. Which rows are
        # merged is a function of the trusted lists only, so G15 invariance is unchanged; with no duplicates the
        # batch is exactly the old one.
        uniq: dict[tuple, int] = {}
        index = [uniq.setdefault(tuple(s), len(uniq)) for s in trusted_ids]
        ulists = [list(u) for u in uniq]
        nonempty = [s if s else [self.pad_id] for s in ulists]
        cids, cmask = pad_batch(nonempty, self.pad_id, device=dev)
        for b, s in enumerate(ulists):
            if not s:
                cmask[b, 0] = False
        self.tokens_processed += int(cmask.sum())
        self._mode = "record"
        try:
            final = self._run(cids, cmask)
        finally:
            self._mode = None
        rec = list(self._rec)
        if len(ulists) != len(trusted_ids):
            sel = torch.tensor(index, dtype=torch.long, device=dev)
            rec = [x.index_select(0, sel) for x in rec]
            final = final.index_select(0, sel)
        self.p_rows_deduplicated += len(trusted_ids) - len(ulists)
        self.last_protected_compact = {"trusted_ids": trusted_ids, "layers": rec, "final": final}
        layers = [carry(rec[i], cnt) for i in range(n_layers)]
        self._rec = [None] * n_layers
        return layers, carry(final, cnt)

    def features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                 trust: torch.Tensor | None = None, p_cache: dict | None = None) -> torch.Tensor:
        if trust is None:
            raise ValueError("TrustGatedBackbone.features needs explicit per-token trust labels (fail closed)")
        self._check_no_checkpointing()
        dev = self.device
        ids = input_ids.to(dev)
        mask = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.to(dev).bool()
        trust = trust.to(dev).bool()
        p_layers, p_final = self.protected_stream(ids, mask, trust, p_cache)
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

    def fork_streams(self, prefix: Sequence[int], conts: Sequence[Sequence[int]], trust: Sequence[bool],
                     cont_trust: Sequence[Sequence[bool]], p_cache: dict | None = None):
        """Two-stream CACHE fork: ``(h_D prefix [T, d], h_P fork [K, d], h_D fork [K, d])``.

        P runs on the K full sequences ``prefix + cont_k`` exactly as the recompute fork does (so the protected
        features, and G15, are those of the recompute path; identical trusted lists run once). The D stream runs
        the prefix ONCE, copies its recurrent state K times (the base backbone's cache fork) and extends each
        copy by its continuation, with the P -> D bridges injecting the carried P features at every position of
        both calls. Equal to the recompute fork up to float rounding (tested); D costs T + sum|cont_k| tokens
        instead of K (T + |cont|)."""
        if trust is None or cont_trust is None:
            raise ValueError("fork_streams needs trust labels for the prefix and every continuation")
        self._check_no_checkpointing()
        dev = self.device
        n, k = len(prefix), len(conts)
        seqs = [list(prefix) + list(c) for c in conts]
        trs = [list(trust) + list(ct) for ct in cont_trust]
        ids, mask = pad_batch(seqs, self.pad_id, device=dev)
        tmask, _ = pad_batch([[int(x) for x in t] for t in trs], 0, device=dev)
        p_layers, p_final = self.protected_stream(ids, mask.bool(), tmask.bool(), p_cache)
        self.last_protected, self.last_protected_final = p_layers, p_final
        last = mask.sum(-1) - 1
        hP = p_final[torch.arange(k, device=dev), last]
        if self.readout == "protected":
            return p_final[0, :n], hP, hP
        self._inject_seq = [[pl[:1, :n] for pl in p_layers], [pl[:, n:] for pl in p_layers]]
        self._calls = [0] * len(self._blocks)
        self._mode = "inject_seq"
        self.tokens_processed += n + sum(len(c) for c in conts)
        try:
            hpre, hD = HiddenStateBackbone.fork_features(self, prefix, conts)
            if any(c != 2 for c in self._calls):
                raise RuntimeError(f"trust-gate cache fork expects one prefix call and one continuation call per "
                                   f"layer, got {self._calls} (this backbone's fork path is not supported)")
        finally:
            self._mode = None
            self._inject_seq = None
        return hpre, hP, hD

    def fork_features(self, prefix: Sequence[int], conts: Sequence[Sequence[int]], trust: Sequence[bool] | None = None,
                      cont_trust: Sequence[Sequence[bool]] | None = None):
        """Fork. ``trust`` labels the prefix and ``cont_trust`` each continuation (both required: option ids are
        data under the provenance/caller policies). ``stream_fork="recompute"`` (default): K full rows, exact.
        ``stream_fork="cache"`` AND ``fork_mode="cache"``: ``fork_streams`` (one D prefix pass + K state copies), so
        ``run_cell``'s smoke stage (which toggles ``fork_mode``) still compares the cache fork against recompute."""
        if trust is None or cont_trust is None:
            raise ValueError("fork_features needs trust labels for the prefix and every continuation")
        if self.stream_fork == "cache" and self.fork_mode == "cache":
            hpre, hP, hD = self.fork_streams(prefix, conts, trust, cont_trust)
            if self.readout == "concat":
                n = len(prefix)
                return (torch.cat([hpre, self.last_protected_final[0, :n].to(hpre.dtype)], -1),
                        torch.cat([hD, hP.to(hD.dtype)], -1))
            return hpre, hD
        tr = list(trust)
        seqs = [list(prefix)] + [list(prefix) + list(c) for c in conts]
        trs = [tr] + [tr + list(ct) for ct in cont_trust]
        ids, mask = pad_batch(seqs, self.pad_id, device=self.device)
        tmask, _ = pad_batch([[int(x) for x in t] for t in trs], 0, device=self.device)
        h = self.features(ids, mask, tmask.bool())
        last = mask.sum(-1) - 1
        return h[0, : len(prefix)], h[torch.arange(1, len(seqs), device=self.device), last[1:]]


def wrap(backbone: HiddenStateBackbone, cfg: dict | None) -> HiddenStateBackbone:
    """Config-flag entry point (train/ and safety_eval): ``{"trust_gate": true, "bridge_rank": 8,
    "readout": "decision", "trust_policy": "provenance", "stream_fork": "recompute" | "cache"}``; anything falsy returns the backbone unchanged."""
    if not cfg or not cfg.get("trust_gate"):
        return backbone
    return TrustGatedBackbone(backbone, bridge_rank=int(cfg.get("bridge_rank", 8)),
                              readout=cfg.get("readout", "decision"), trust_policy=cfg.get("trust_policy", DEFAULT_POLICY),
                              seed=int(cfg.get("seed", 0)), stream_fork=cfg.get("stream_fork", "recompute"))


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
