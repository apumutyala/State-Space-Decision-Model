"""Letter-logit decider over a stock HF causal LM (smoke/stand-in parent; the real adapter lives in decide/).

Scoring recipe ported from chl-typed-control/tc/score.py (the prior lab's verified recipe):
  * one forward pass per (request, option order); next-token logits restricted to the option letters, each
    letter = logsumexp over its single-token spellings ("A", " A"); fp32 logits from the final hidden state;
  * RIGHT padding and no padding mask (causal mask => pads cannot affect real tokens; left padding in bf16 moved
    probabilities by up to 0.12 in the prior lab);
  * two option orders (identity and a fixed derangement), probabilities averaged (position debiasing);
  * letter mass (letters' share of the full-vocabulary mass) reported as a format-compliance diagnostic.

Untrusted state fields are sanitised (safety_eval.sanitize) before rendering unless ``sanitize=False`` (a
control arm only), and the sanitised span is checked at token level (fails closed).

The prompt puts the state before the question and options, so text before an attacker slot inside the state is
identical across option orders (GCG-lite relies on this; ``render(..., slot=...)``).
"""
from __future__ import annotations

import time

import torch

from .request import allowed_options, renormalise, untrusted_fields
from .sanitize import assert_no_special_ids, sanitize_text, tokenizer_markers

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = "You are a typed decision engine. Choose exactly one option."


def perm_order(n: int, perm: int) -> list[int]:
    """perm 0: identity; perm 1: rotate by n//2 then reverse (a derangement for n >= 2: no index fixed)."""
    if perm == 0:
        return list(range(n))
    rot = [(i + n // 2) % n for i in range(n)]
    out = rot[::-1]
    if any(out[i] == i for i in range(n)):  # fall back to a plain rotation by 1 (always a derangement)
        out = [(i + 1) % n for i in range(n)]
    return out


class HFLetterDecider:
    def __init__(self, model_name: str, device: str = "cuda", dtype: str = "bfloat16", sanitize: bool = True,
                 mask: bool = True, max_tokens: int = 16384, max_batch: int = 32, revision: str | None = None,
                 local_files_only: bool = True, tok=None, model=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name, self.device, self.sanitize, self.mask = model_name, device, sanitize, mask
        self.tok = tok or AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only)
        self.tok.padding_side = "right"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=getattr(torch, dtype), local_files_only=local_files_only,
            device_map=device if device != "cpu" else None)
        self.model.eval()
        self.max_tokens, self.max_batch = max_tokens, max_batch
        self.markers = tokenizer_markers(self.tok)
        self.groups = self._letter_groups()
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.thinking_flag = "qwen3" in model_name.lower()

    # ------------------------------------------------------------ rendering
    def _letter_groups(self, n: int = len(LETTERS)) -> list[list[int]]:
        out = []
        for L in LETTERS[:n]:
            ids = set()
            for s in (L, " " + L):
                t = self.tok.encode(s, add_special_tokens=False)
                if len(t) == 1:
                    ids.add(t[0])
            out.append(sorted(ids))
        flat = [i for g in out for i in g]
        assert len(flat) == len(set(flat)), "letter token ids collide"
        return out

    def prepare(self, request: dict) -> tuple[dict, int]:
        """Sanitise untrusted fields (unless disabled); returns (request copy, markers broken)."""
        if not self.sanitize:
            return request, 0
        import copy
        r = copy.deepcopy(request)
        n = 0
        for f in untrusted_fields(r):
            if isinstance(r["state"].get(f), str):
                r["state"][f], k = sanitize_text(r["state"][f], self.markers)
                assert_no_special_ids(self.tok, r["state"][f])
                n += k
        return r, n

    def user_text(self, request: dict, qkey: str, perm: int, slot: tuple[str, str] | None = None) -> str:
        st = request["state"]
        unt = set(untrusted_fields(request))
        lines = ["State:"]
        if isinstance(st, dict):
            for f, v in st.items():
                tag = "third-party content" if f in unt else "trusted"
                txt = v if isinstance(v, str) else str(v)
                if slot is not None and f == slot[0]:
                    txt = txt + slot[1]
                lines.append(f"[{f} | {tag}]\n{txt}")
        else:
            lines.append(str(st))
        q = request["questions"][qkey]
        keys = list(q["criteria"])
        order = perm_order(len(keys), perm)
        lines += ["", "Question:", str(q.get("instructions", "")), "", "Options:"]
        for j, i in enumerate(order):
            lines.append(f"{LETTERS[j]}. {keys[i]}: {q['criteria'][keys[i]]}")
        lines += ["", "Answer with the letter of the chosen option only."]
        return "\n".join(lines)

    def render(self, request: dict, qkey: str, perm: int, slot: tuple[str, str] | None = None) -> str:
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": self.user_text(request, qkey, perm, slot)}]
        kw = {"tokenize": False, "add_generation_prompt": True}
        if self.thinking_flag:
            kw["enable_thinking"] = False
        return self.tok.apply_chat_template(msgs, **kw)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------ scoring
    def letter_logits_from_hidden(self, h: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B, H] final-norm hidden -> (letter log-probs [B, n] fp32, log letter-mass share [B])."""
        W = self.model.get_output_embeddings().weight
        full = h.float() @ W.float().t() if h.shape[0] * W.shape[0] <= 64 * 2 ** 20 else None
        per = []
        for g in self.groups[:n]:
            per.append(torch.logsumexp(h.float() @ W[g].float().t(), dim=-1))
        per = torch.stack(per, dim=-1)
        lp = per - torch.logsumexp(per, dim=-1, keepdim=True)
        mass = (torch.logsumexp(per, -1) - torch.logsumexp(full, -1)) if full is not None else torch.full_like(lp[:, 0], float("nan"))
        return lp, mass

    def hidden_last(self, ids: torch.Tensor, lens: list[int]) -> torch.Tensor:
        out = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        rows = torch.arange(len(lens), device=ids.device)
        return out[rows, torch.tensor(lens, device=ids.device) - 1]

    @torch.no_grad()
    def score_ids(self, encs: list[list[int]], ns: list[int]) -> list[tuple[list[float], float]]:
        order = sorted(range(len(encs)), key=lambda i: -len(encs[i]))
        dev = next(self.model.parameters()).device
        res: list = [None] * len(encs)
        k = 0
        while k < len(order):
            L = len(encs[order[k]])
            bs = max(1, min(self.max_batch, self.max_tokens // max(L, 1)))
            idx = order[k:k + bs]
            k += bs
            ids = torch.full((len(idx), L), self.tok.pad_token_id, dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, :len(encs[i])] = torch.tensor(encs[i])
            h = self.hidden_last(ids.to(dev), [len(encs[i]) for i in idx])
            nmax = max(ns[i] for i in idx)
            lp, mass = self.letter_logits_from_hidden(h, nmax)
            for r, i in enumerate(idx):
                l = lp[r, :ns[i]]
                l = l - torch.logsumexp(l, -1)
                res[i] = (l.tolist(), float(mass[r]))
        return res

    def decide_many(self, requests: list[dict]) -> list:
        t0 = time.time()
        jobs, prepared, out = [], [], []
        for ri, req in enumerate(requests):
            try:
                r, nsan = self.prepare(req)
            except Exception as e:  # noqa: BLE001 - fail closed on residual special tokens
                prepared.append(e)
                continue
            prepared.append((r, nsan))
            for q, spec in r["questions"].items():
                n = len(spec["criteria"])
                if n > len(LETTERS):
                    raise ValueError("more options than letters")
                for p in (0, 1):
                    jobs.append((ri, q, p, self.encode(self.render(r, q, p)), n))
        scores = self.score_ids([j[3] for j in jobs], [j[4] for j in jobs])
        acc: dict = {}
        for (ri, q, p, enc, n), (lp, mass) in zip(jobs, scores):
            acc.setdefault((ri, q), []).append((p, lp, mass, len(enc)))
        per_req = (time.time() - t0) * 1000 / max(len(requests), 1)
        for ri, req in enumerate(requests):
            if isinstance(prepared[ri], Exception):
                out.append(prepared[ri])
                continue
            r, nsan = prepared[ri]
            answers, raw_all, pc_all, masses, ntok = {}, {}, {}, [], []
            for q, spec in r["questions"].items():
                keys = list(spec["criteria"])
                P = torch.zeros(len(keys), dtype=torch.float64)
                for p, lp, mass, nt in acc[(ri, q)]:
                    order = perm_order(len(keys), p)
                    pr = torch.tensor(lp, dtype=torch.float64).exp()
                    for j, i in enumerate(order):
                        P[i] += 0.5 * pr[j]
                    masses.append(mass)
                    ntok.append(nt)
                raw = {k: float(P[i]) for i, k in enumerate(keys)}
                pc = renormalise(raw, allowed_options(r, q))
                served = pc if self.mask else raw
                answers[q] = {"type": "choice", "choice": max(served, key=served.get), "probabilities": served}
                raw_all[q], pc_all[q] = raw, pc
            out.append({"status": "ok", "answers": answers, "raw_probabilities": raw_all,
                        "policy_conditioned_probabilities": pc_all,
                        "diagnostics": {"model_revision": f"{self.name}@{self.revision}", "markers_sanitised": nsan,
                                        "sanitize": self.sanitize, "mask": self.mask,
                                        "letter_mass_log_min": min(masses), "input_tokens": ntok,
                                        "approx_ms_per_request": round(per_req, 2)}})
        return out

    def __call__(self, request: dict) -> dict:
        r = self.decide_many([request])[0]
        if isinstance(r, Exception):
            raise r
        return r
