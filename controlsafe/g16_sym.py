"""G16-sym: the G16 protected judge made POSITION-FREE, for the reorder/relabel-invariant ``sym-fork`` layout.

Why a new head. ``controlsafe.g16.G16Head.judge`` adds per-slot parameters (``a_k = A h_P[k] + u_k``,
``b_k = w . h_P[k] + beta_k``) and decides with ``torch.argmax`` (first index wins ties). Both break condition (C2)
/ (C4) of docs/guarantees/REORDER_INVARIANCE.md. The slot parameters existed because under the ``provenance`` policy
the option text is data, so ``h_P[k]`` coincides across options and only the slot could separate them.

Construction. Rows are ``sym-fork`` rows (``heads.symmetric``): the shared prefix holds state + question + the NUMBER
of options; option k is a continuation holding its description only (no id), and the option axis is content-canonical.
For each option k, with ONE shared parameter set (nothing indexed by k):

* ``h_P[k]``: G15 protected features at the end of ``prefix + continuation_k``. The option description is TRUSTED iff
  the host declared its field ``/questions/<qid>/criteria/<id>`` instruction-trusted in ``request["ssdm"]
  ["provenance"]`` (``row.meta["trusted_paths"]``; fail closed otherwise), or the caller opts in with
  ``option_trust="host_schema"`` (every description is host schema). So ``h_P[k]`` is a function of (trusted prefix,
  option k's trusted content) only.
* ``a_k = A h_P[k]``, ``b_k = w . h_P[k]``; ``E_k = c tanh(W_e h_D[k])`` in ``(-c, c)^r``; ``score_k = b_k + a_k . E_k``.
* Trusted-only interval: ``lo_k = b_k - c ||a_k||_1``, ``hi_k = b_k + c ||a_k||_1`` (float64, rounded OUTWARD so the
  float32 ``score_k`` always lies inside, assumption A5 of INJECTION_ROBUST_CONFORMAL.md).
* Host veto keyed by option ID (it travels with the option); vetoed options have ``lo = hi = score = -inf`` and are
  listed separately in the export.
* Decision: the argmax SET (exact float equality). A single representative is chosen by the SHA-256 of the option's
  continuation token ids (= its content); if that does not resolve the tie (duplicate contents) the representative
  is ``None`` and only the set is reported. Never by position, never by id.

Guarantees (stated with their assumptions):

1. **Reorder + relabel (C1-C4).** Every per-option quantity (lo, hi, score) is a function of (prefix, option content),
   the normaliser is a set softmax, ties are content-resolved. With the content-canonical sym-fork axis the row, hence
   every float op, is identical for any reorder / relabel: outputs are BITWISE equal (tests/guarantees harness, tol=0).
   Relabel needs the descriptions to be non-empty (``row.meta["id_as_content"]`` lists the exceptions).
2. **G15 on the intervals.** ``lo``, ``hi`` (and the veto) are bitwise invariant to untrusted content and lengths;
   so is the exported ``act`` (built from them alone). ``decision_set`` / ``decision`` = argmax of ``b + a.E`` read
   E, i.e. untrusted text: they are NOT G15-invariant (reorder/relabel-invariant only). Under a BPE tokenizer this
   relies on sym-fork v3's trust-segment tokenization (no token straddles a trust boundary, heads.symmetric).
3. **G16 (a)-(d)** of ``controlsafe/g16.py`` carry over unchanged (the judge is the same function minus u/beta).
4. **Injection-robust sets.** Any conformal score built from (lo, hi) alone is attack-proof (Theorem A); the matched
   ``s_lo`` calibration (``matched_qhat``) is exact split conformal; act only on the trusted-only singleton
   (``act``), never on the attacked argmax.

Scope. If no option description is trusted (the policy default for undeclared criteria), ``h_P[k]`` coincides across
options, all intervals are equal and the robust set is everything: honest, not a bug. Under ``trust_policy="state"``
the continuation (descriptions) is trusted by that policy; under ``"caller"`` it never is.
Export schema: docs/G16_EXPORT.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from heads.backbone import pad_batch

from .g16 import EvidenceChannel, independent_of_untrusted, radii
from .trust_gate import ID_PATH, TrustGatedBackbone, continuation_trust, row_trust

GUARANTEE_ID = "ssdm.g16sym.position_free_judge.v1"
EXPORT_SCHEMA = "ssdm.g16sym.export.v1"
OPTION_TRUST = ("declared", "host_schema")
SYM_FORK_LAYOUTS = ("sym-fork", "sym-fork-chat", "sym-fork-ids")
_U32 = 2.0 ** -24  # fp32 unit roundoff


# --------------------------------------------------------------------------------------------- trust labels
def _criteria_path(qid: str, oid: str) -> str:
    from decide.provenance import criteria_path

    return criteria_path(qid, oid)


def option_content_trusted(row, policy: str, option_trust: str = "declared") -> list[bool]:
    """Per option (row axis order): is its description trusted? A function of (option id -> host declaration), so it
    travels with the option under reorder and with the host's relabeled provenance under relabel."""
    if option_trust not in OPTION_TRUST:
        raise ValueError(f"option_trust must be one of {OPTION_TRUST}")
    meta = getattr(row, "meta", {}) or {}
    no_desc = set(meta.get("id_as_content", []))  # rendered by its id: the id is data, never trusted
    if policy == "caller":
        return [False] * row.n_options
    if policy == "state":
        return [True] * row.n_options
    if option_trust == "host_schema":
        return [o not in no_desc for o in row.option_ids]
    ok = set(meta.get("trusted_paths", []))
    return [o not in no_desc and _criteria_path(row.qid, o) in ok for o in row.option_ids]


def sym_continuation_trust(row, k: int, policy: str, option_trust: str = "declared") -> list[bool]:
    """Trust of continuation k: template tokens trusted, the description span trusted iff the option's content is."""
    if policy in ("state", "caller"):
        return continuation_trust(row, k, policy)
    n = len(row.continuations[k])
    t = [True] * n
    trusted = option_content_trusted(row, policy, option_trust)[k]
    for a, b, path in row.meta["cont_spans"][k]:
        if path != ID_PATH:
            raise ValueError(f"unexpected continuation span path {path!r} (expected sym-fork rows)")
        if not trusted:
            for i in range(a, min(b, n)):
                t[i] = False
    return t


def content_hash(cont_ids: Sequence[int]) -> str:
    return hashlib.sha256(json.dumps([int(x) for x in cont_ids]).encode()).hexdigest()


# --------------------------------------------------------------------------------------------- head
@dataclass
class G16SymOutput:
    option_ids: list[str]
    score: torch.Tensor  # [K] float32 point logits (-inf on vetoed)
    lo: torch.Tensor  # [K] float64, outward-rounded (-inf on vetoed)
    hi: torch.Tensor  # [K] float64
    a: torch.Tensor  # [K, r]
    b: torch.Tensor  # [K]
    E: torch.Tensor  # [K, r]
    allowed: torch.Tensor  # [K] bool
    content_sha: list[str]
    decision_set: list[str] = field(default_factory=list)
    decision: str | None = None

    def probs(self) -> dict[str, float]:
        """Set softmax in float64 with an exactly rounded (order-free) normaliser."""
        z = [float(v) for v in self.score.double()]
        m = max(v for v in z if v != -math.inf)
        e = [math.exp(v - m) if v != -math.inf else 0.0 for v in z]
        s = math.fsum(e)
        return {o: v / s for o, v in zip(self.option_ids, e)}

    def intervals(self) -> dict[str, list[float]]:
        """Allowed options only (vetoed ones can never be the decision): {id: [lo, hi]}."""
        return {o: [float(l), float(h)] for o, l, h, m in zip(self.option_ids, self.lo, self.hi, self.allowed) if m}


def decide_set(option_ids: Sequence[str], score: torch.Tensor, allowed: torch.Tensor,
               content_sha: Sequence[str]) -> tuple[list[str], str | None]:
    """(argmax set by id, content-hash representative or None). Exact float ties; no position, no id ordering."""
    z = score.double()
    m = max(float(v) for v, ok in zip(z, allowed) if ok)
    idx = [k for k in range(len(z)) if bool(allowed[k]) and float(z[k]) == m]
    best = min(content_sha[k] for k in idx)
    winners = [k for k in idx if content_sha[k] == best]
    rep = option_ids[winners[0]] if len(winners) == 1 else None
    return sorted(option_ids[k] for k in idx), rep


def outward_interval(a: torch.Tensor, b: torch.Tensor, c: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``b -/+ c ||a||_1`` in float64, widened by a bound on the float32 rounding of ``b + a.E`` (r+2 ops)."""
    a64, b64 = a.double(), b.double()
    w = c * a64.abs().sum(-1)
    r = a.shape[-1]
    slack = (r + 4) * _U32 * (b64.abs() + w) + 1e-30
    lo, hi = b64 - w - slack, b64 + w + slack
    return torch.nextafter(lo, torch.full_like(lo, -math.inf)), torch.nextafter(hi, torch.full_like(hi, math.inf))


class G16SymHead(nn.Module):
    """Position-free G16 judge over sym-fork rows. ``forward(row, host_allowed_ids=None) -> G16SymOutput``."""

    name = "g16sym"
    wants_rows = True
    attention_free = True

    def __init__(self, gated: TrustGatedBackbone, r: int = 8, c: float = 1.0, option_trust: str = "declared",
                 seed: int = 0, stream_mode: str = "recompute", p_stream_per_row: bool = True) -> None:
        super().__init__()
        if gated.readout != "decision":
            raise ValueError("G16-sym needs the gated backbone's D stream (readout='decision')")
        if option_trust not in OPTION_TRUST:
            raise ValueError(f"option_trust must be one of {OPTION_TRUST}")
        if stream_mode not in ("recompute", "cache"):
            raise ValueError(stream_mode)
        self.gated, self.option_trust, self.stream_mode = gated, option_trust, stream_mode
        #: run the P stream one distinct trusted list at a time (see ``_p_cache_per_row``); False = one batched call
        self.p_stream_per_row = bool(p_stream_per_row)
        d = gated.model.config.hidden_size
        g = torch.Generator().manual_seed(seed * 104729 + 5)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=g)))
            self.evidence = EvidenceChannel(d, r, c)
            self.A = nn.Linear(d, r, bias=False)
            self.w = nn.Linear(d, 1, bias=False)
        self.config = {"r": r, "c": float(c), "option_trust": option_trust, "stream_mode": stream_mode,
                       "p_stream_per_row": self.p_stream_per_row}

    @property
    def c(self) -> float:
        return self.evidence.c

    def own_parameters(self) -> list[nn.Parameter]:
        bb = {id(p) for p in self.gated.parameters()}
        return [p for p in self.parameters() if id(p) not in bb]

    # -------------------------------------------------------------- streams
    def _check_row(self, row) -> None:
        if getattr(row, "layout", None) not in SYM_FORK_LAYOUTS:
            raise ValueError(f"G16-sym needs sym-fork rows, got layout {getattr(row, 'layout', None)!r}")

    def streams_unique(self, row) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """(h_P [U, d], h_D [U, d], idx): the streams of each DISTINCT (continuation tokens, trust mask), and for each
        option on the row axis the index of its distinct row. Exact duplicates are computed ONCE and copied: a
        batched forward rounds by batch row, so two identical rows in one batch can differ in the last bits (SYM_CHECK
        round 2: E of duplicates up to 1.5e-5 apart at hidden 128, so a relabel among duplicates changed the served
        decision). Copying makes duplicates bitwise equal by construction (and saves their compute)."""
        self._check_row(row)
        pol = self.gated.trust_policy
        tr = row_trust(row, pol)
        ctr = [sym_continuation_trust(row, k, pol, self.option_trust) for k in range(row.n_options)]
        conts = row.continuations
        full = [(tuple(int(t) for t in c), tuple(bool(x) for x in ct)) for c, ct in zip(conts, ctr)]
        # Batch order = a function of the TRUSTED token lists first (so an untrusted edit cannot reorder the P batch:
        # G15 is bitwise only for a host-controlled batch shape), then of the full key (content: reorder/relabel-free)
        tkey = {k: tuple(t for t, ok in zip(list(row.ids) + list(k[0]), tr + list(k[1])) if ok) for k in set(full)}
        uniq = sorted(set(full), key=lambda k: (tkey[k], k))
        pos = {k: i for i, k in enumerate(uniq)}
        idx = [pos[k] for k in full]
        conts_u, ctr_u = [list(k[0]) for k in uniq], [list(k[1]) for k in uniq]
        seqs = [list(row.ids) + c for c in conts_u]
        ids, mask = pad_batch(seqs, self.gated.pad_id, device=self.gated.device)
        tmask, _ = pad_batch([[int(x) for x in tr + t] for t in ctr_u], 0, device=self.gated.device)
        p_cache = self._p_cache_per_row(ids, mask, tmask) if self.p_stream_per_row else None
        if self.stream_mode == "cache":
            _, hP, hD = self.gated.fork_streams(row.ids, conts_u, tr, ctr_u, p_cache=p_cache)
            return hP.float(), hD.float(), idx
        hD = self.gated.features(ids, mask, tmask.bool(), p_cache=p_cache)
        hP = self.gated.last_protected_final
        last = mask.sum(-1) - 1
        ar = torch.arange(ids.shape[0], device=ids.device)
        return hP[ar, last].float(), hD[ar, last].float(), idx

    def _p_cache_per_row(self, ids: torch.Tensor, mask: torch.Tensor, tmask: torch.Tensor) -> dict:
        """P stream run ONE distinct trusted token list at a time (batch of 1), assembled into the gated backbone's
        ``p_cache`` format so the batched call reuses it exactly. With merging, untrusted text decides how many
        distinct rows the batch has and where each sits; per row, h_P of an option is a function of its own trusted
        list alone, whatever the batch (no batch-invariance assumption on the P kernels, e.g. on GPU)."""
        from .trust_gate import compaction

        g = self.gated
        pos, _ = compaction(mask.bool(), tmask.bool())
        trusted = [[int(ids[b, j]) for j in p] for b, p in enumerate(pos)]
        done: dict[tuple, tuple[list[torch.Tensor], torch.Tensor]] = {}
        for s in trusted:
            key = tuple(s)
            if key in done or not s:
                continue
            one = torch.tensor([s], dtype=torch.long, device=ids.device)
            ones = torch.ones_like(one, dtype=torch.bool)
            g.protected_stream(one, ones, ones)
            c = g.last_protected_compact
            done[key] = (list(c["layers"]), c["final"])
        if not done:
            return None
        proto_layers, proto_final = next(iter(done.values()))
        tc = max(len(s) for s in trusted)

        def pad(x: torch.Tensor | None, like: torch.Tensor) -> torch.Tensor:
            out = like.new_zeros((1, tc) + tuple(like.shape[2:]))
            if x is not None:
                out[:, : x.shape[1]] = x
            return out

        layers = [torch.cat([pad(done[tuple(s)][0][i] if s else None, proto_layers[i]) for s in trusted], 0)
                  for i in range(len(proto_layers))]
        final = torch.cat([pad(done[tuple(s)][1] if s else None, proto_final) for s in trusted], 0)
        return {"trusted_ids": trusted, "layers": layers, "final": final}

    def streams(self, row) -> tuple[torch.Tensor, torch.Tensor]:
        """(h_P [K, d], h_D [K, d]) at the end of each option continuation (row axis order)."""
        hP, hD, idx = self.streams_unique(row)
        sel = torch.tensor(idx, dtype=torch.long)
        return hP[sel], hD[sel]

    # -------------------------------------------------------------- judge (trusted-only)
    def judge(self, hP: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(a [K, r], b [K]) from P features only: the SAME map for every option, no slot index anywhere.

        Row by row (K one-row products), not one [K, d] GEMM: a CPU GEMM's rounding can depend on a row's POSITION in
        the batch (blocked kernels, remainder rows), and the axis position of a trusted option moves when an UNTRUSTED
        description sorts elsewhere. Measured on cpu-8 / torch 2.13 (job-8kqt9): lo moved by 2.4e-7 under undeclared
        description edits, and two duplicate options got lo values 2.4e-7 apart. Per row, each option's (a, b) is a
        function of its own h_P row alone, so G15 and the duplicate tie hold bitwise whatever the batch."""
        a = torch.cat([self.A(hP[k:k + 1]) for k in range(hP.shape[0])], 0)
        b = torch.cat([self.w(hP[k:k + 1]) for k in range(hP.shape[0])], 0).squeeze(-1)
        return a, b

    def forward(self, row, host_allowed_ids: Iterable[str] | None = None) -> G16SymOutput:
        hP_u, hD_u, idx = self.streams_unique(row)
        sel = torch.tensor(idx, dtype=torch.long)
        a_u, b_u = self.judge(hP_u)
        E_u = torch.cat([self.evidence(hD_u[i:i + 1]) for i in range(hD_u.shape[0])], 0)  # per row, as the judge
        a, b, E = a_u[sel], b_u[sel], E_u[sel]  # duplicates: copies of one computation, bitwise equal
        allowed = torch.tensor([True] * row.n_options if host_allowed_ids is None else
                               [o in set(host_allowed_ids) for o in row.option_ids], dtype=torch.bool)
        if not bool(allowed.any()):
            raise ValueError("no allowed option (host allowed set is empty)")
        score = b + (a * E).sum(-1)
        score = torch.where(allowed, score, torch.full_like(score, -math.inf))
        lo, hi = outward_interval(a, b, self.c)
        ninf = torch.full_like(lo, -math.inf)
        lo, hi = torch.where(allowed, lo, ninf), torch.where(allowed, hi, ninf)
        sha = [content_hash(c) for c in row.continuations]
        dset, rep = decide_set(row.option_ids, score, allowed, sha)
        return G16SymOutput(list(row.option_ids), score, lo, hi, a, b, E, allowed, sha, dset, rep)

    def certificate(self, out: G16SymOutput) -> dict[str, Any]:
        """G16 (c)/(d) fields on the position-free judge (radius w.r.t. E; independence over the box)."""
        rad = radii(out.a, out.score, out.allowed)
        return {"schema": GUARANTEE_ID, **rad, "decision_set": out.decision_set, "decision": out.decision,
                "independent_of_untrusted": independent_of_untrusted(out.a, out.b, out.allowed, self.c)}


# --------------------------------------------------------------------------------------------- export
def export_record(decision_id: str, out: G16SymOutput, gold: str | None = None,
                  extra: dict[str, Any] | None = None, qhat: float | None = None,
                  kappa: float | None = None) -> dict[str, Any]:
    """One JSONL record (docs/G16_EXPORT.md). ``options`` holds the allowed options only; vetoed ids are listed.

    ``act`` is THE injection-robust action: the trusted-only singleton of the matched robust set built from [lo, hi]
    alone (``act(rec, qhat, kappa)``; None = escalate, and None when no calibrated ``qhat`` is given). It is
    G15-invariant. ``decision_set`` / ``decision`` are the point argmax of ``score = b + a.E``; E reads untrusted text,
    so they are NOT G15-invariant (reorder/relabel-invariant only) and must never drive a guaranteed action."""
    opts = []
    for k, o in enumerate(out.option_ids):
        if not bool(out.allowed[k]):
            continue
        lo, hi, s = float(out.lo[k]), float(out.hi[k]), float(out.score[k])
        if not (math.isfinite(lo) and math.isfinite(hi) and lo <= s <= hi):
            raise ValueError(f"option {o!r}: non-finite or score outside [lo, hi] ({lo}, {s}, {hi}); refusing to export")
        opts.append({"id": o, "lo": lo, "hi": hi, "score": s, "content_sha256": out.content_sha[k]})
    rec: dict[str, Any] = {"schema": EXPORT_SCHEMA, "decision_id": str(decision_id), "options": opts,
                           "vetoed": [o for o, m in zip(out.option_ids, out.allowed.tolist()) if not m],
                           "decision_set": out.decision_set, "decision": out.decision}
    if gold is not None:
        rec["gold"] = gold
    rec["act"] = None if qhat is None else act(rec, qhat, kappa)
    rec["act_rule"] = None if qhat is None else {"qhat": float(qhat), "kappa": kappa}
    if extra:
        rec.update({k: v for k, v in extra.items() if k not in rec})
    return rec


def validate_record(rec: dict[str, Any]) -> dict[str, list[float]]:
    """Schema check; returns {id: [lo, hi]} (the shape jevkit.guarantees.robust_set consumes)."""
    if rec.get("schema") != EXPORT_SCHEMA:
        raise ValueError(f"schema must be {EXPORT_SCHEMA!r}")
    if not isinstance(rec.get("decision_id"), str) or not rec.get("options"):
        raise ValueError("decision_id (str) and a non-empty options list are required")
    iv: dict[str, list[float]] = {}
    for o in rec["options"]:
        oid, lo, hi, s = o["id"], float(o["lo"]), float(o["hi"]), float(o["score"])
        if oid in iv or not (math.isfinite(lo) and math.isfinite(hi)) or not lo <= s <= hi:
            raise ValueError(f"bad option {oid!r}")
        iv[oid] = [lo, hi]
    if "gold" in rec and rec["gold"] not in iv and rec["gold"] not in rec.get("vetoed", []):
        raise ValueError("gold must be an option id")
    if rec.get("act") is not None:  # the exported action must be the trusted-only robust singleton, nothing else
        rule = rec.get("act_rule")
        if not isinstance(rule, dict) or "qhat" not in rule:
            raise ValueError("act needs act_rule {qhat, kappa}")
        if rec["act"] != act({k: v for k, v in rec.items() if k not in ("act", "act_rule")}, float(rule["qhat"]),
                             rule.get("kappa")):
            raise ValueError("act is not the matched robust singleton of [lo, hi]")
    return iv


def write_jsonl(path, records: Iterable[dict[str, Any]]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            validate_record(r)
            f.write(json.dumps(r, sort_keys=True, allow_nan=False) + "\n")
            n += 1
    return n


def read_jsonl(path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        recs = [json.loads(line) for line in f if line.strip()]
    for r in recs:
        validate_record(r)
    return recs


# --------------------------------------------------------------------------------------------- robust sets (matched)
def matched_qhat(cal: Sequence[dict[str, Any]], alpha: float) -> float:
    """Theorem A / Corollary A': q-hat = Q_alpha of s_lo(intervals_i, gold_i) over calibration records. Needs only
    [lo, hi] + gold. A gold that was vetoed scores +inf (it can never be covered; counted, not dropped)."""
    from jevkit.guarantees import conformal_qhat, robust_min_score

    s = []
    for r in cal:
        iv = validate_record(r)
        s.append(robust_min_score(iv, r["gold"]) if r["gold"] in iv else math.inf)
    return conformal_qhat(s, alpha)


def robust_set(rec: dict[str, Any], qhat: float) -> list[str]:
    from jevkit.guarantees import robust_set as rs

    return rs(validate_record(rec), qhat)


def act(rec: dict[str, Any], qhat: float, kappa: float | None = None) -> str | None:
    """Corollary E: act iff the matched robust set is a singleton (and, optionally, the singleton's smallest
    probability over the box >= kappa). Returns that singleton -- never the attacked argmax -- or None (escalate)."""
    cs = robust_set(rec, qhat)
    if len(cs) != 1:
        return None
    y = cs[0]
    if kappa is not None:
        iv = validate_record(rec)
        z = {k: (v[0] if k == y else v[1]) for k, v in iv.items()}
        m = max(z.values())
        e = {k: math.exp(v - m) for k, v in z.items()}
        if e[y] / math.fsum(e.values()) < kappa:
            return None
    return y


# --------------------------------------------------------------------------------------------- training losses
def g16sym_losses(out: G16SymOutput, gold: str, c: float, width_coef: float = 0.0,
                  robust_coef: float = 0.0) -> dict[str, torch.Tensor]:
    """Loss terms for one decision (docs/G16_EXPORT.md, "Training"). All differentiable in A, w, W_e and the LoRA:

    * ``ce``: cross-entropy of softmax(score) (the point head; trains everything incl. E).
    * ``robust_ce``: cross-entropy at the worst box corner (gold at ``lo``, rivals at ``hi``): an upper bound on the
      CE under ANY untrusted text, so it trains the trusted margin ``b_y - b_k > c(||a_y||_1 + ||a_k||_1)`` that makes
      the matched robust set small / a singleton. Closed form: no extra forward pass.
    * ``width``: mean ``c ||a_k||_1`` over allowed options (interval half-width). The penalty trades evidence use
      (E can only move a logit by this much) against robust-set size.
    """
    k = out.option_ids.index(gold)
    m, b = out.allowed, out.b
    ce = -torch.log_softmax(out.score, -1)[k]
    w1 = out.a.abs().sum(-1)
    z_worst = torch.where(torch.arange(len(b)) == k, b - c * w1, b + c * w1)
    z_worst = torch.where(m, z_worst, torch.full_like(z_worst, -math.inf))
    robust = -torch.log_softmax(z_worst, -1)[k]
    width = (c * w1)[m].mean()
    total = ce + robust_coef * robust + width_coef * width
    return {"total": total, "ce": ce, "robust_ce": robust, "width": width}
