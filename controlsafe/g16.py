"""G16 (optional arm): a protected-only judge with one bounded evidence channel, a trusted structural veto, a
certified radius, and G4 run over the evidence channel only. Design: PR #42 review (comment 6020811487),
built on the G15 trust gate.

For one (request, question) row with options k = 1..K (ordered by the schema's canonical option order):

* ``h_P[k]`` -- G15 protected features at the end of ``prefix + continuation_k`` (trusted tokens only).
* ``h_D[k]`` -- the data stream (full context) at the same position.
* **Evidence channel** ``E_k = c * tanh(W_e h_D[k])`` in ``(-c, c)^r``. ``W_e`` is zero-initialised. This
  is the ONLY quantity computed from the data stream that the decision reads.
* **Judge** ``a_k = A h_P[k] + u_k`` (in R^r) and ``b_k = w . h_P[k] + beta_k``. ``u_k`` and ``beta_k`` are
  per-option-position parameters: under the provenance policy the option ids are data, so the h_P[k] can
  coincide across k.
* **Trusted veto** ``m_k`` in {0,1}: the host's allowed set (``request["policy"]["allowed"][qid]``, host-only
  per G2(a)), ANDed with an optional P-only learned gate ``sigmoid(g . h_P[k]) > 1/2``.
* ``logit_k = b_k + a_k . E_k`` for ``m_k = 1``; ``-inf`` otherwise. The decision is ``i = argmax_k logit_k``.

**G16 guarantee** (exact; under the G15 trust labels and treating the schema, i.e. the option set, its
canonical order and the allowed set, as host structure):

(a) **Veto, STRUCTURAL.** No untrusted text can make a vetoed option the decision. ``m`` is a function of
    host fields and P only, and vetoed logits are ``-inf``.
(b) **Only E reaches the decision, STRUCTURAL.** ``a``, ``b`` and ``m`` are bitwise invariant to untrusted
    content and lengths (G15). The decision is therefore a fixed, known function of ``E``, which is bounded.
(c) **Certified radius, CERTIFIED per decision.** Let ``mu_ij = logit_i - logit_j``. No perturbation
    ``E -> E'`` with ``||E' - E||_2 < rho2 = min_j mu_ij / sqrt(||a_i||^2 + ||a_j||^2)`` (joint L2 over all
    options) can change the decision. Per-option balls ``max_k ||E'_k - E_k||_2 < rho_opt =
    min_j mu_ij / (||a_i|| + ||a_j||)`` cannot change it either. Both bounds are tight (tested).
    NOTE: the review's ``mu / L`` with ``L = max_k ||a_k||`` is NOT sound in either norm (it can exceed rho2
    by up to sqrt(2) and rho_opt by up to 2; ``tests/controlsafe/test_g16.py`` has the counterexample).
    ``mu / (sqrt(2) L) <= rho2`` is a sound simplification.
(d) **Independence, CERTIFIED per decision.** If for every allowed j != i
    ``b_i - b_j > c (||a_i||_1 + ||a_j||_1)``, then the decision is the same for EVERY value of E in the box,
    hence for every untrusted text: ``independent_of_untrusted = True``. G4 holds for any k at zero recompute.
(e) **G4 over E, CERTIFIED (the exact erase-and-check certificate).** Otherwise erase-and-check recomputes
    only E for each origin. P is reused exactly: the origins share the trusted subsequence, and reuse is
    checked token by token. The decision on each origin equals a full recompute of the G16 model (tested).

Scope: E is computed from the D stream, so it can still be steered within its box by untrusted text. The
guarantee is that steering is bounded, logged (``E`` is r*K numbers per decision) and certified-or-escalated,
not that it is impossible. The capability cost against the D-head arm is ``trust_gate.capability_cost``.
Training: ``G16DecisionHead`` is the ``train/`` head arm (``TrainConfig(head="g16")`` over a trust-gated backbone,
``DecisionModel`` dispatches rows to ``forward_rows``). The decision is ``softmax(logit)`` for EVERY question type:
the cumulative-link ordinal output of the other heads would be a second decision function outside the certificate,
so it is not used (losses, incl. RPS on score questions, read these log-probabilities). ``stream_mode="cache"``
computes the two streams with ``TrustGatedBackbone.fork_streams`` (one D prefix pass + K state copies; P once per
distinct trusted subsequence) instead of K full recomputes; both modes give the same decision function up to float
rounding, and ``certify`` uses whichever the head was built with. Snapshot reuse of the D stream inside G4-over-E is
not implemented (each origin's D stream is recomputed; P is not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from heads.backbone import pad_batch

from .trust_gate import TrustGatedBackbone, continuation_trust, row_trust

GUARANTEE_ID = "ssdm.g16.protected_judge_bounded_evidence.v1"
STREAM_MODES = ("recompute", "cache")


class EvidenceChannel(nn.Module):
    """``E = c * tanh(W_e h)`` in ``(-c, c)^r``; ``W_e`` zero-init (E == 0 at init)."""

    def __init__(self, d: int, r: int = 8, c: float = 1.0) -> None:
        super().__init__()
        if c <= 0 or r < 1:
            raise ValueError((r, c))
        self.c, self.r = float(c), r
        self.W_e = nn.Linear(d, r, bias=False)
        nn.init.zeros_(self.W_e.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.c * torch.tanh(self.W_e(h.float()))


@dataclass
class G16Output:
    logits: torch.Tensor  # [K] (-inf on vetoed options)
    E: torch.Tensor  # [K, r]
    a: torch.Tensor  # [K, r]
    b: torch.Tensor  # [K]
    allowed: torch.Tensor  # [K] bool (host AND P-gate)
    decision: int


class G16Head(nn.Module):
    def __init__(self, gated: TrustGatedBackbone, r: int = 8, c: float = 1.0, max_options: int = 256,
                 p_gate: bool = False, seed: int = 0, stream_mode: str = "recompute") -> None:
        super().__init__()
        if stream_mode not in STREAM_MODES:
            raise ValueError(f"stream_mode must be one of {STREAM_MODES}, got {stream_mode!r}")
        self.stream_mode = stream_mode
        if gated.readout != "decision":
            raise ValueError("G16 needs the gated backbone's D stream (readout='decision')")
        self.gated = gated
        d = gated.model.config.hidden_size
        g = torch.Generator().manual_seed(seed * 104729 + 3)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            self.evidence = EvidenceChannel(d, r, c)
            self.A = nn.Linear(d, r, bias=False)
            self.w = nn.Linear(d, 1, bias=False)
            self.gate = nn.Linear(d, 1) if p_gate else None
        self.u = nn.Parameter(torch.zeros(max_options, r))
        self.beta = nn.Parameter(torch.zeros(max_options))
        self.max_options = max_options

    # ------------------------------------------------------------------ inputs
    def _batch(self, row, policy: str):
        tr = row_trust(row, policy)
        seqs = [list(row.ids) + list(c) for c in row.continuations]
        trs = [tr + continuation_trust(row, k, policy) for k in range(len(seqs))]
        ids, mask = pad_batch(seqs, self.gated.pad_id, device=self.gated.device)
        tmask, _ = pad_batch([[int(x) for x in t] for t in trs], 0, device=self.gated.device)
        return ids, mask, tmask.bool()

    def streams(self, row, p_cache: dict | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """(h_P [K, d], h_D [K, d]) at the end of each option continuation."""
        if self.stream_mode == "cache":
            pol = self.gated.trust_policy
            ctr = [continuation_trust(row, k, pol) for k in range(len(row.continuations))]
            _, hP, hD = self.gated.fork_streams(row.ids, row.continuations, row_trust(row, pol), ctr, p_cache=p_cache)
            return hP.float(), hD.float()
        ids, mask, trust = self._batch(row, self.gated.trust_policy)
        hD = self.gated.features(ids, mask, trust, p_cache=p_cache)
        hP = self.gated.last_protected_final
        last = mask.sum(-1) - 1
        ar = torch.arange(ids.shape[0], device=ids.device)
        return hP[ar, last].float(), hD[ar, last].float()

    def judge(self, hP: torch.Tensor, host_allowed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(a [K, r], b [K], allowed [K]) from P features, option positions and host fields ONLY."""
        K = hP.shape[0]
        if K > self.max_options:
            raise ValueError(f"{K} options > max_options {self.max_options}")
        a = self.A(hP) + self.u[:K]
        b = self.w(hP).squeeze(-1) + self.beta[:K]
        allowed = host_allowed.bool().to(hP.device)
        if self.gate is not None:
            allowed = allowed & (torch.sigmoid(self.gate(hP).squeeze(-1)) > 0.5)
        if not bool(allowed.any()):
            raise ValueError("no allowed option (host allowed set AND P-gate is empty)")
        return a, b, allowed

    @staticmethod
    def combine(a: torch.Tensor, b: torch.Tensor, E: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        logits = b + (a * E).sum(-1)
        return torch.where(allowed, logits, torch.full_like(logits, float("-inf")))

    def forward(self, row, host_allowed: torch.Tensor | None = None) -> G16Output:
        hP, hD = self.streams(row)
        allowed_in = torch.ones(hP.shape[0], dtype=torch.bool) if host_allowed is None else host_allowed
        a, b, allowed = self.judge(hP, allowed_in)
        E = self.evidence(hD)
        logits = self.combine(a, b, E, allowed)
        return G16Output(logits, E, a, b, allowed, int(torch.argmax(logits)))

    @staticmethod
    def log_probs(out: G16Output) -> torch.Tensor:
        return torch.log_softmax(out.logits, dim=-1)


def host_allowed_ids(request: dict, qid: str) -> set[str] | None:
    """Host allowed set for ``qid``: ``request["policy"]["allowed"][qid]`` (safety_eval shape) and/or the decide
    envelope's ``request["ssdm"]["policy_restriction"][qid]`` (host-only, validated by the adapter); both present
    -> their intersection; neither -> None (all allowed)."""
    sets = []
    pol = ((request.get("policy") or {}).get("allowed") or {})
    if qid in pol:
        sets.append(set(pol[qid]))
    env = request.get("ssdm") if isinstance(request.get("ssdm"), dict) else {}
    res = env.get("policy_restriction") or {}
    if isinstance(res, dict) and qid in res:
        sets.append(set(res[qid]))
    if not sets:
        return None
    return set.intersection(*sets)


def host_allowed_mask(request: dict, qid: str, option_ids: Sequence[str]) -> torch.Tensor:
    """Host allowed set (``host_allowed_ids``; absent -> all allowed), in row order."""
    ok = host_allowed_ids(request, qid)
    if ok is None:
        return torch.ones(len(option_ids), dtype=torch.bool)
    return torch.tensor([o in ok for o in option_ids], dtype=torch.bool)


def attach_host_allowed(row, request: dict) -> None:
    """Stamp the host allowed set on a ``LayoutRow`` (``row.meta["host_allowed"]``), read by ``forward_rows``."""
    ok = host_allowed_ids(request, row.qid)
    row.meta["host_allowed"] = None if ok is None else sorted(ok)


def row_allowed_mask(row) -> torch.Tensor:
    ok = (getattr(row, "meta", {}) or {}).get("host_allowed")
    if ok is None:
        return torch.ones(row.n_options, dtype=torch.bool)
    ok = set(ok)
    return torch.tensor([o in ok for o in row.option_ids], dtype=torch.bool)


class G16DecisionHead(G16Head):
    """G16 as a ``train/`` head arm (``DecisionModel(gated_backbone, "g16", head_kwargs)``).

    ``forward_rows(rows)`` returns per-option logits ``[B, Kmax]`` (``-inf`` on vetoed and padded options); the
    per-row ``G16Output`` (a, b, E, allowed) of the last call is kept in ``last_outputs`` for the certificate fields.
    The gated backbone is the DecisionModel's own backbone (shared, not copied): ``DecisionModel.head_parameters``
    excludes its parameters, which train in the LoRA group (LoRA + P -> D bridges)."""

    name = "g16"
    attention_free = True
    needs_fork = False
    wants_rows = True

    def __init__(self, gated: TrustGatedBackbone, r: int = 8, c: float = 1.0, max_options: int = 256,
                 p_gate: bool = False, seed: int = 0, stream_mode: str = "cache") -> None:
        super().__init__(gated, r=r, c=c, max_options=max_options, p_gate=p_gate, seed=seed, stream_mode=stream_mode)
        self.config = {"r": r, "c": float(c), "max_options": max_options, "p_gate": p_gate, "stream_mode": stream_mode}
        self.last_outputs: list[G16Output] = []

    def own_parameters(self) -> list[nn.Parameter]:
        bb = {id(p) for p in self.gated.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.own_parameters())

    def forward_rows(self, rows) -> Any:
        from heads.heads import HeadOutput

        outs = []
        for row in rows:
            hP, hD = self.streams(row)
            a, b, allowed = self.judge(hP, row_allowed_mask(row))
            E = self.evidence(hD)
            logits = self.combine(a, b, E, allowed)
            outs.append(G16Output(logits, E, a, b, allowed, int(torch.argmax(logits))))
        self.last_outputs = outs
        kmax = max(len(o.logits) for o in outs)
        z = torch.stack([torch.cat([o.logits, o.logits.new_full((kmax - len(o.logits),), float("-inf"))])
                         for o in outs])
        return HeadOutput(z)

    def certificate(self, i: int = 0) -> dict[str, Any]:
        """Per-decision certificate fields of row ``i`` of the last ``forward_rows`` (no recompute)."""
        o = self.last_outputs[i]
        rad = radii(o.a, o.logits, o.allowed)
        return {"schema": GUARANTEE_ID, **rad, "independent_of_untrusted": independent_of_untrusted(
                    o.a, o.b, o.allowed, self.evidence.c),
                "radius_covers_box": radius_covers_box(rad["rho_joint_l2"], o.E, o.allowed, self.evidence.c),
                "c": self.evidence.c, "E": o.E.tolist(), "allowed_mask": o.allowed.tolist()}

    def cost(self, T: int, K: int, trusted_frac: float = 0.5, cont_len: int = 8) -> dict:
        d, r = self.A.in_features, self.A.out_features
        return {"flops": 2 * K * (2 * d * r + d) + 2 * K * r, "accumulator_scalars": K * (2 * r + 1),
                "attention_free": True, "streaming_implemented": False,
                "backbone_tokens_per_row": (1 + trusted_frac) * T + K * cont_len if self.stream_mode == "cache"
                else K * (1 + trusted_frac) * (T + cont_len),
                "note": "two streams: D over all T tokens + P over the trusted ones (trusted_frac assumed)"}


def radius_covers_box(rho_joint: float, E: torch.Tensor, allowed: torch.Tensor, c: float) -> bool:
    """Sound radius certificate against ANY untrusted text: the joint radius exceeds the largest L2 distance from
    E to any point of the box ``[-c, c]^{K x r}`` (restricted to allowed options; vetoed ones never matter)."""
    Ed = E.double()[allowed.bool()]
    far = torch.maximum((c - Ed).abs(), (c + Ed).abs())
    return bool(rho_joint > float(far.pow(2).sum().sqrt()))


def certificate_summary(records: Sequence[dict]) -> dict[str, Any]:
    """Eval-card stats over ``certify`` records: certified vs escalated, by reason, radii, independence, cost."""
    import collections
    import statistics

    n = len(records)
    if n == 0:
        return {"n": 0}
    fin = [x["rho_joint_l2"] for x in records if x.get("rho_joint_l2") is not None and x["rho_joint_l2"] != float("inf")]
    fo = [x["rho_per_option_l2"] for x in records if x.get("rho_per_option_l2") not in (None, float("inf"))]
    cert = sum(bool(x.get("certified")) for x in records)
    return {"n": n, "certified_rate": cert / n, "escalated_rate": 1 - cert / n,
            "independence_rate": sum(bool(x.get("independent_of_untrusted")) for x in records) / n,
            "radius_covers_box_rate": sum(bool(x.get("radius_covers_box")) for x in records) / n,
            "reasons": dict(collections.Counter(str(x.get("reason") or "g4_over_E_certified") for x in records)),
            "rho_joint_l2_mean": statistics.fmean(fin) if fin else None,
            "rho_joint_l2_median": statistics.median(fin) if fin else None,
            "rho_per_option_l2_mean": statistics.fmean(fo) if fo else None,
            "margin_mean": statistics.fmean([x["margin"] for x in records if x.get("margin") not in (None, float("inf"))])
            if any(x.get("margin") not in (None, float("inf")) for x in records) else None,
            "origins_evaluated": sum(x.get("n_origins_evaluated", 0) for x in records),
            "p_cache_misses": sum(x.get("p_cache_misses", 0) for x in records),
            "d_tokens": sum(x.get("d_tokens", 0) for x in records)}


# ----------------------------------------------------------------------------------- certificates
def radii(a: torch.Tensor, logits: torch.Tensor, allowed: torch.Tensor) -> dict[str, float]:
    """Certified radii of the decision w.r.t. E (see module docstring (c)), plus the unsound ``mu / L``."""
    a = a.double()
    lg = logits.double()
    i = int(torch.argmax(lg))
    others = [j for j in range(len(lg)) if j != i and bool(allowed[j])]
    if not others:
        inf = float("inf")
        return {"decision": i, "margin": inf, "rho_joint_l2": inf, "rho_per_option_l2": inf, "mu_over_L": inf,
                "lipschitz_L": float(a.norm(dim=-1).max())}
    n = a.norm(dim=-1)
    mu = torch.stack([lg[i] - lg[j] for j in others])
    nj = torch.stack([n[j] for j in others])
    rho2 = mu / torch.sqrt(n[i] ** 2 + nj ** 2).clamp_min(1e-300)
    rhoo = mu / (n[i] + nj).clamp_min(1e-300)
    L = float(n[[i] + others].max())
    return {"decision": i, "margin": float(mu.min()), "rho_joint_l2": float(rho2.min()),
            "rho_per_option_l2": float(rhoo.min()), "lipschitz_L": L,
            "mu_over_L": float(mu.min()) / L if L > 0 else float("inf"),
            "mu_over_sqrt2_L": float(mu.min()) / (L * 2 ** 0.5) if L > 0 else float("inf")}


def independent_of_untrusted(a: torch.Tensor, b: torch.Tensor, allowed: torch.Tensor, c: float) -> bool:
    """Module docstring (d): the argmax of ``b + a . E`` is the same for every E in ``[-c, c]^{K x r}``."""
    a, b = a.double(), b.double()
    idx = [k for k in range(len(b)) if bool(allowed[k])]
    l1 = a.abs().sum(-1)
    # worst case for option i: E_i minimises a_i.E_i (-c||a_i||_1), every rival maximises (+c||a_j||_1)
    i = max(idx, key=lambda k: float(b[k] - c * l1[k]))
    return all(float(b[i] - b[j]) > c * float(l1[i] + l1[j]) for j in idx if j != i)


@torch.no_grad()
def certify(head: G16Head, request: dict, qid: str, k: int, build_row: Callable[[dict], Any],
            verify: bool = False) -> dict[str, Any]:
    """G16 decision + certificate for one request. ``build_row(request) -> LayoutRow`` is the real layout
    (``heads.layout.build_row(req, qid, encode)``). Origins come from ``safety_eval.certify`` (covered
    insertions, same segmenter/dedup). With ``verify`` every origin is ALSO run through the full G16 model
    from scratch and the decisions must match."""
    from safety_eval import certify as C

    gated = head.gated
    row = build_row(request)
    hP, hD = head.streams(row)
    cache = gated.last_protected_compact
    allowed_in = host_allowed_mask(request, qid, row.option_ids)
    a, b, allowed = head.judge(hP, allowed_in)
    E = head.evidence(hD)
    logits = head.combine(a, b, E, allowed)
    rad = radii(a, logits, allowed)
    indep = independent_of_untrusted(a, b, allowed, head.evidence.c)
    try:  # fail closed: anything that cannot be evaluated is not a radius certificate
        cov = bool(torch.isfinite(E).all()) and radius_covers_box(rad["rho_joint_l2"], E, allowed, head.evidence.c)
    except (KeyError, IndexError, RuntimeError, TypeError):
        cov = False
    rad = {**rad, "radius_covers_box": cov}
    rec: dict[str, Any] = {"schema": GUARANTEE_ID, **rad, "decision_index": rad["decision"],
                           "decision": row.option_ids[rad["decision"]],
                           "independent_of_untrusted": indep, "E": E.tolist(), "k": k,
                           "allowed": [o for o, m in zip(row.option_ids, allowed.tolist()) if m]}
    if not (torch.isfinite(E).all() and torch.isfinite(logits).all()):
        # Non-finite evidence or logits: never certify (review round 2, non-blocking -> fixed).
        rec.update({"certified": False, "reason": "non_finite_evidence_or_logits", "n_origins_evaluated": 0,
                    "d_tokens": 0, "p_cache_misses": 0})
        return rec
    if indep:
        rec.update({"certified": True, "reason": "independent_of_untrusted", "n_origins_evaluated": 0,
                    "d_tokens": 0, "p_cache_misses": 0})
        return rec
    # B3 (review round 2): erasure origins only edit the state. Untrusted text anywhere else (e.g. an injected
    # passage placed in question instructions) cannot be erased, so a certificate over origins would be vacuous
    # for it: escalate instead. Option ids are host structure (excluded).
    from heads.layout import ID_PATH, state_path
    from controlsafe.trust_gate import DEFAULT_POLICY
    meta = getattr(row, "meta", {}) or {}
    trusted_paths = set(meta.get("trusted_paths", []))
    outside = sorted({p for _, _, p in meta.get("field_spans", [])
                      if p not in trusted_paths and p != ID_PATH and p != state_path()})
    if outside:
        rec.update({"certified": False, "reason": "untrusted_outside_erasable_fields", "untrusted_paths": outside,
                    "n_origins_evaluated": 0, "d_tokens": 0, "p_cache_misses": 0, "trust_policy": DEFAULT_POLICY})
        return rec
    if not any(p == state_path() and p not in trusted_paths for _, _, p in meta.get("field_spans", [])):
        # No untrusted text at all (besides host-structure option ids): the decision cannot depend on it.
        rec.update({"certified": True, "reason": "no_untrusted_text", "n_origins_evaluated": 0,
                    "d_tokens": 0, "p_cache_misses": 0})
        return rec
    hits0 = gated.p_cache_hits
    changed, d_tokens, n = [], 0, 0
    seen = {C.request_hash(request)}
    for lab, r in C.origin_requests(request, k):
        h = C.request_hash(r)
        if h in seen:
            continue
        seen.add(h)
        n += 1
        ro = build_row(r)
        if ro.option_ids != row.option_ids:
            raise ValueError("origin changed the option set: not a covered insertion")
        hPo, hDo = head.streams(ro, p_cache=cache)
        if not torch.equal(hPo, hP):  # G15 says this cannot happen; check it rather than assume it
            raise AssertionError("protected features differ across origins: trust labels inconsistent")
        d_tokens += sum(len(ro.ids) + len(c) for c in ro.continuations)
        lo = head.combine(a, b, head.evidence(hDo), allowed)
        do = int(torch.argmax(lo))
        if verify:
            full = head(ro, allowed_in)
            if full.decision != do:
                raise AssertionError("G4-over-E decision differs from the full G16 recompute")
        if do != rad["decision"]:
            changed.append({"erased": list(lab), "decision": ro.option_ids[do]})
    if n == 0:
        # No erasable origin was evaluated: nothing was checked, so do not certify (B3).
        rec.update({"certified": False, "reason": "no_origins_evaluated", "n_origins_evaluated": 0,
                    "changed": [], "d_tokens": 0, "p_cache_misses": 0})
        return rec
    rec.update({"certified": not changed, "reason": None if not changed else "decision_changed_under_erasure",
                "n_origins_evaluated": n, "changed": changed, "d_tokens": d_tokens,
                "p_cache_misses": n - (gated.p_cache_hits - hits0)})
    return rec
