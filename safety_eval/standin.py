"""CPU stand-in decider for tests and pipeline checks (NOT a model; numbers from it mean nothing).

``KeywordDecider`` scores each option by word overlap between the state and the option description, plus a
cue bonus for options whose distinctive tokens sit in a sentence with "due/outstanding/..." words, plus a
deliberate gullibility term: every mention of an option key (or its distinctive words) inside an *imperative*
sentence of the state adds ``gullibility``. That makes hijack and steer measurable on it, so tests can check
the metric plumbing end to end. It honours the interface: raw, policy-conditioned and served probabilities.
"""
from __future__ import annotations

import math
import re

from .request import allowed_options, renormalise

STOP = {"the", "a", "an", "to", "of", "for", "and", "or", "is", "was", "on", "in", "my", "your", "this", "that",
        "it", "be", "with", "from", "by", "route", "customer", "customer's", "has", "not", "yet", "ask", "pay",
        "invoice", "user", "any", "without", "before"}
CUE = re.compile(r"\b(due|outstanding|open|needs paying|still needs|payable now|unpaid)\b", re.I)
IMPERATIVE = re.compile(r"\b(must|select|choose|instead|please|should|right step|correct next|authoris|"
                        r"override|do not use|ignore)\w*", re.I)


def words(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9\-']+", s.lower()) if w not in STOP and len(w) > 2}


def sentences(s: str) -> list[str]:
    return [x for x in re.split(r"(?<=[.!?\]])\s+|\n+", s) if x.strip()]


class KeywordDecider:
    def __init__(self, gullibility: float = 3.0, temperature: float = 1.0, mask: bool = True):
        self.g, self.t, self.mask = gullibility, temperature, mask
        self.calls = 0

    def __call__(self, request: dict) -> dict:
        self.calls += 1
        st = request["state"]
        text = "\n".join(v for v in st.values() if isinstance(v, str)) if isinstance(st, dict) else str(st)
        sw = words(text)
        sents = sentences(text)
        answers, raw_all, pc_all = {}, {}, {}
        for q, spec in request["questions"].items():
            crit = spec["criteria"]
            scores = {}
            for k, desc in crit.items():
                dw = words(desc)
                ident = {w for w in dw if re.search(r"\d", w)} or dw
                s = len(sw & dw) / (1 + len(dw)) * 4
                for sen in sents:
                    sl = words(sen)
                    hit = (k.lower() in sen.lower()) or (ident and len(ident & sl) >= max(1, len(ident) // 2))
                    if hit and CUE.search(sen):
                        s += 2.0
                    if hit and IMPERATIVE.search(sen):
                        s += self.g
                scores[k] = s / self.t
            mx = max(scores.values())
            z = sum(math.exp(v - mx) for v in scores.values())
            raw = {k: math.exp(v - mx) / z for k, v in scores.items()}
            allowed = allowed_options(request, q)
            pc = renormalise(raw, allowed)
            served = pc if self.mask else raw
            choice = max(served, key=served.get)
            answers[q] = {"type": "choice", "choice": choice, "probabilities": served}
            raw_all[q], pc_all[q] = raw, pc
        return {"status": "ok", "answers": answers, "raw_probabilities": raw_all,
                "policy_conditioned_probabilities": pc_all,
                "diagnostics": {"model_revision": "standin-keyword-v1"}}
