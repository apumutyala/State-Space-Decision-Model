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
Training: ``log_probs`` gives a loss-ready output. A ``train/`` head arm is not wired yet. Snapshot reuse of
the D stream inside G4-over-E is not implemented (each origin's D stream is recomputed; P is not).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
from torch import nn

from heads.backbone import pad_batch

from .trust_gate import TrustGatedBackbone, continuation_trust, row_trust

GUARANTEE_ID = "ssdm.g16.protected_judge_bounded_evidence.v1"


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
                 p_gate: bool = False, seed: int = 0) -> None:
        super().__init__()
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


def host_allowed_mask(request: dict, qid: str, option_ids: Sequence[str]) -> torch.Tensor:
    """Host allowed set (``request["policy"]["allowed"][qid]``; absent -> all allowed), in row order."""
    pol = ((request.get("policy") or {}).get("allowed") or {})
    if qid not in pol:
        return torch.ones(len(option_ids), dtype=torch.bool)
    ok = set(pol[qid])
    return torch.tensor([o in ok for o in option_ids], dtype=torch.bool)


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
    rec: dict[str, Any] = {"schema": GUARANTEE_ID, **rad, "decision_index": rad["decision"],
                           "decision": row.option_ids[rad["decision"]],
                           "independent_of_untrusted": indep, "E": E.tolist(), "k": k,
                           "allowed": [o for o, m in zip(row.option_ids, allowed.tolist()) if m]}
    if indep:
        rec.update({"certified": True, "reason": "independent_of_untrusted", "n_origins_evaluated": 0,
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
    rec.update({"certified": not changed, "reason": None if not changed else "decision_changed_under_erasure",
                "n_origins_evaluated": n, "changed": changed, "d_tokens": d_tokens,
                "p_cache_misses": n - (gated.p_cache_hits - hits0)})
    return rec
