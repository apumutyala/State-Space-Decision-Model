"""GCG-lite against Clef's joint schema head (SAFETY_SPEC G6, PREREG 2.3), same budget and objectives as
``safety_eval/gcg.py`` but with Clef's own readout instead of a letter readout.

What carries over unchanged from ``safety_eval.gcg``: ``HP_GATE``/``HP_SMOKE``, ``objective_loss``,
``order_outcome``, ``sample_candidates``, ``GUARD``/``SLOT``, the token allow-list (plus ``"`` and ``\\``,
which JSON-escaping of the state would otherwise rewrite). Success is re-checked afterwards through the
unmodified ``decide`` path (``ClefBackend`` + adapter) on the decoded string, as for every other model.

Clef-specific facts:
* Clef sorts choice options by id, so it has ONE option order. To reuse the [C, 2, n] machinery the single
  order is duplicated; ``pooled`` and ``both_orders`` are therefore identical for Clef by construction, and
  the receipt says so instead of implying an order-robustness result.
* The adversarial tokens sit inside the JSON-rendered ``state``. Gradients flow through the backbone and the
  joint head (lexical option vectors index the output embedding by option-span ids, which never include the
  attack tokens). All model parameters are frozen (``requires_grad = False``) so 27B fits for backward.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import torch

from safety_eval.data import QKEY, strip_meta
from safety_eval.gcg import GUARD, SLOT, objective_loss, order_outcome, outcome, sample_candidates

from .clef_backend import ClefBackend


def allowed_tokens(tok: Any, vocab_rows: int) -> torch.Tensor:
    ok = torch.zeros(vocab_rows, dtype=torch.bool)
    special = set(tok.all_special_ids) | set(getattr(tok, "added_tokens_decoder", {}).keys())
    for i in range(min(vocab_rows, len(tok))):
        if i in special:
            continue
        s = tok.decode([i])
        ok[i] = bool(s) and s.isascii() and s.isprintable() and not any(c in s for c in '<>|"\\')
    return ok


def _text_model(clef_model: Any):
    base = clef_model.language_model
    base = base.get_base_model() if hasattr(base, "get_base_model") else base
    tm = base.model
    if hasattr(tm, "language_model"):
        tm = tm.language_model
    return base, tm


class ClefProblem:
    """One request + objective; the attack slot is appended to the untrusted state field."""

    def __init__(self, backend: ClefBackend, request: dict, objective: str):
        m = request["_meta"]
        self.request, self.objective, self.meta = request, objective, m
        self.req = strip_meta(request)
        self.field = m["untrusted_field"]
        self.clean_value = self.req["state"][self.field]
        enc = backend.encoder
        slotted = copy.deepcopy(self.req)
        slotted["state"][self.field] = self.clean_value + SLOT + GUARD
        pre_text, post_text = backend.jsm.render(slotted["state"]).split(SLOT)
        self.parts = enc.parts(self.req)  # schema/prefix/suffix (state replaced per candidate)
        self.pre_state = enc.untrusted_ids(pre_text)
        self.post_state = enc.untrusted_ids(post_text)
        q = self.req["questions"][QKEY]
        self.keys = list(q["criteria"])  # request order (= safety_eval key order)
        self.n = len(self.keys)
        self.qindex = [qq.question_id for qq in self.parts.questions].index(QKEY)
        oids = self.parts.questions[self.qindex].option_ids
        self.to_key = [oids.index(k) for k in self.keys]  # key position -> Clef option position
        tkey = m["steer_target"] if objective == "steer" else m["attacker_option"]
        self.target = self.keys.index(tkey)
        self.gold = self.keys.index(m["gold"])

    def state_ids(self, adv: list[int]) -> list[int]:
        return self.pre_state + list(adv) + self.post_state

    def with_suffix(self, suffix_text: str) -> dict:
        r = copy.deepcopy(self.req)
        r["state"][self.field] = self.clean_value + suffix_text
        return r


class ClefGCGEngine:
    def __init__(self, backend: ClefBackend, eval_batch: int = 16):
        self.be, self.model, self.tok = backend, backend.model, backend.tok
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.base, self.text = _text_model(self.model)
        self.E = self.text.get_input_embeddings().weight
        self.dev = self.E.device
        self.eval_batch = eval_batch
        self.n_seqs = 0

    def _lp_keys(self, prob: ClefProblem, q_logits: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(q_logits.float(), dim=-1)[..., prob.to_key]
        return torch.stack([lp, lp], dim=-2)  # one order, duplicated (see module docstring)

    @torch.no_grad()
    def eval_advs(self, prob: ClefProblem, advs: torch.Tensor) -> torch.Tensor:
        """advs [C, n_adv] -> key-order log-probs [C, 2, n] (fp32)."""
        out = []
        for i in range(0, advs.shape[0], self.eval_batch):
            recs = [self.be.encoder.assemble(prob.parts, prob.state_ids(a.tolist())) for a in advs[i:i + self.eval_batch]]
            res = self.be.logits_for_records(recs)
            self.n_seqs += len(recs)
            out += [self._lp_keys(prob, r[prob.qindex]) for r in res]
        return torch.stack(out).cpu()

    def loss_grad(self, probs: list[ClefProblem], adv: torch.Tensor) -> tuple[float, torch.Tensor]:
        one = torch.zeros(adv.shape[0], self.E.shape[0], device=self.dev, dtype=self.E.dtype)
        one[torch.arange(adv.shape[0]), adv.to(self.dev)] = 1.0
        one.requires_grad_(True)
        tot = 0.0
        W = self.base.get_output_embeddings().weight
        for prob in probs:
            rec = self.be.encoder.assemble(prob.parts, prob.state_ids(adv.tolist()))
            ids = torch.tensor(rec.input_ids, device=self.dev).unsqueeze(0)
            a0 = len(prob.parts.prefix_ids) + len(prob.pre_state)
            emb = torch.cat([self.E[ids[0, :a0]], one @ self.E, self.E[ids[0, a0 + adv.shape[0]:]]], 0).unsqueeze(0)
            mask = torch.ones_like(ids)
            h = self.text(inputs_embeds=emb, attention_mask=mask, use_cache=False, return_dict=True).last_hidden_state
            ql = self.model.head(h, ids, mask, [rec], W)[0][prob.qindex]
            loss = objective_loss(self._lp_keys(prob, ql), prob.target, prob.objective) / len(probs)
            loss.backward()
            tot += float(loss.detach())
            self.n_seqs += 1
        return tot, one.grad.float()

    def roundtrip_ok(self, prob: ClefProblem, advs: torch.Tensor) -> torch.Tensor:
        """True where the served encoding of the decoded string reproduces pre + adv + post exactly."""
        out = []
        for a in advs:
            r = prob.with_suffix(self.tok.decode(a.tolist()) + GUARD)
            try:
                out.append(self.be.encoder.state_ids(r["state"]) == prob.state_ids(a.tolist()))
            except Exception:  # noqa: BLE001  (fail-closed encoder refusal = not a valid attack string)
                out.append(False)
        return torch.tensor(out)


def attack(eng: ClefGCGEngine, probs: list[ClefProblem], hp: dict, seed: int, search: str = "gcg",
           early_stop: bool = True, log=print) -> dict:
    tok = eng.tok
    gen = torch.Generator().manual_seed(seed)
    init = tok(hp["init"], add_special_tokens=False).input_ids
    assert len(init) == 1, f"init {hp['init']!r} must be one token"
    adv = torch.tensor(init * hp["n_adv"])
    allowed = allowed_tokens(tok, eng.E.shape[0])
    t0, trace, first = time.time(), [], None
    best_loss, best_adv = float("inf"), adv.clone()
    for step in range(hp["steps"]):
        grad = eng.loss_grad(probs, adv)[1] if search == "gcg" else None
        cand = sample_candidates(adv, grad, allowed, hp["batch"], hp["topk"], gen)
        ok = eng.roundtrip_ok(probs[0], cand)
        cand = cand[ok] if ok.any() else cand
        losses = torch.zeros(cand.shape[0])
        lpks = []
        for prob in probs:
            lpk = eng.eval_advs(prob, cand)
            losses += objective_loss(lpk, prob.target, prob.objective) / len(probs)
            lpks.append(lpk)
        j = int(losses.argmin())
        adv = cand[j].clone()
        if float(losses[j]) < best_loss:
            best_loss, best_adv = float(losses[j]), adv.clone()
        trace.append(round(float(losses[j]), 5))
        if len(probs) == 1 and early_stop and outcome(probs[0], lpks[0][j])["success"]:
            first, best_adv = step + 1, adv.clone()
            break
        if step % 10 == 0:
            log(f"[clef-gcg] step {step} loss {float(losses[j]):.4f} best {best_loss:.4f} cands {cand.shape[0]} "
                f"({time.time() - t0:.0f}s)")
    text = tok.decode(best_adv.tolist())
    return {"adv_ids": best_adv.tolist(), "adv_text": text, "suffix": text + GUARD, "loss_trace": trace,
            "best_loss": best_loss, "steps_run": len(trace), "first_success_step": first,
            "secs": round(time.time() - t0, 2), "n_seqs": eng.n_seqs, "search": search, "hp": hp, "seed": seed,
            "orders": "clef has one option order (sorted ids); pooled == both_orders by construction"}


def final_outcome(eng: ClefGCGEngine, prob: ClefProblem, adv_ids: list[int]) -> dict:
    return order_outcome(prob, eng.eval_advs(prob, torch.tensor([adv_ids]))[0])
